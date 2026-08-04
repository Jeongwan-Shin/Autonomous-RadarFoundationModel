"""Coordinate transforms and derived radar features.

Three frames are in play and mixing them silently is the easiest way to ruin a
training run:

  sensor  radar-native spherical (azimuth, elevation, distance)
  rig     ego-centric at a given timestamp. `obstacle.offline` boxes live here,
          with `reference_frame='rig'` and reference timestamp equal to their
          own timestamp.
  world   clip-local, anchored at the ego pose at t=0 with zero yaw.
          `egomotion.offline` lives here.

A stationary object is stationary in *world*. In rig it appears to move at
roughly the ego speed, so judging motion in rig gets the answer backwards:
measured on 5,000 tracks, rig- and world-frame displacement correlate at only
0.194, and 63.9% of tracks that are genuinely stationary look like movers in
rig.

Radar-to-rig transforms need `calibration/sensor_extrinsics/` (the non-offline
variant). The `.offline` variant carries cameras and lidar only -- no radar.
With the real extrinsics applied, box centres sit a median 1.31 m from their
nearest radar return, which is the expected vehicle-scale agreement.
"""

import numpy as np
from scipy.spatial.transform import Rotation


def spherical_to_sensor_xyz(azimuth, elevation, distance):
    """Radar spherical -> Cartesian in the sensor frame. Angles in radians."""
    ce = np.cos(elevation)
    return np.stack([distance * ce * np.cos(azimuth),
                     distance * ce * np.sin(azimuth),
                     distance * np.sin(elevation)], axis=-1)


def extrinsics_to_matrix(row):
    """One row of sensor_extrinsics (qx,qy,qz,qw,x,y,z) -> (R, t)."""
    rot = Rotation.from_quat([row["qx"], row["qy"], row["qz"], row["qw"]]).as_matrix()
    return rot, np.array([row["x"], row["y"], row["z"]], dtype=float)


def sensor_to_rig(points_sensor, rot, translation):
    return points_sensor @ rot.T + translation


def interpolate_pose(timestamps_us, ego_t_us, ego_xyz, ego_quat):
    """Ego pose at arbitrary times.

    Position is linearly interpolated; rotation takes the nearer keyframe.
    At 10 Hz egomotion the residual rotation error over a 50 ms gap is well
    below the radar angular resolution, so slerp is not worth the cost here.
    """
    t = np.asarray(timestamps_us, dtype=float)
    t = np.clip(t, ego_t_us[0], ego_t_us[-1])
    hi = np.clip(np.searchsorted(ego_t_us, t), 1, len(ego_t_us) - 1)
    lo = hi - 1
    span = (ego_t_us[hi] - ego_t_us[lo]).astype(float)
    w = np.where(span > 0, (t - ego_t_us[lo]) / np.where(span > 0, span, 1.0), 0.0)
    xyz = ego_xyz[lo] * (1.0 - w[:, None]) + ego_xyz[hi] * w[:, None]
    nearest = np.where(w < 0.5, lo, hi)
    rot = Rotation.from_quat(ego_quat[nearest]).as_matrix()
    return xyz, rot


def rig_to_world(points_rig, ego_xyz, ego_rot):
    """Per-point transform. points_rig (N,3), ego_xyz (N,3), ego_rot (N,3,3)."""
    return np.einsum("nij,nj->ni", ego_rot, points_rig) + ego_xyz


def ego_velocity(ego_t_us, ego_xyz):
    """World-frame ego velocity [m/s] at each egomotion sample."""
    return np.gradient(ego_xyz, ego_t_us / 1e6, axis=0)


def ego_speed(ego_t_us, ego_xyz):
    return np.linalg.norm(ego_velocity(ego_t_us, ego_xyz)[:, :2], axis=1)


def doppler_residual(points_rig, radial_velocity, ego_vel_rig,
                     doppler_ambiguity=None, max_fold=3):
    """How far each return's Doppler is from what a static world would give.

    A stationary point's radial velocity is the negative projection of the ego
    velocity onto the sensor-to-target ray, so the residual isolates genuine
    object motion. This needs no labels, which is what makes it usable both as
    a pruning criterion and as a free supervision signal.

    Pass `doppler_ambiguity` -- the per-return column of the same name -- or the
    result is meaningless for the mid-range radar. Its unambiguous interval is
    about 22 m/s, so `radial_velocity` saturates around +/-12.7 and a 16 m/s ego
    speed simply cannot be represented: every return then looks like a mover.
    Measured on one MRR scan, undoing the folding drops the residual median from
    22.53 to 0.32 m/s and the fraction above 1 m/s from 100% to 23%. The imaging
    LRR has an interval near 48.6 m/s and is unaffected either way.

    Rotational ego motion (omega x r) is ignored: at 20 Hz the term is small
    next to the ~1 m/s threshold used for pruning.

    On unfolded imaging LRR a 1.0 m/s threshold keeps roughly 47 of 954 returns
    per scan, i.e. it discards 95% of the point cloud.
    """
    ray = points_rig / np.maximum(
        np.linalg.norm(points_rig, axis=1, keepdims=True), 1e-6)
    expected = -(ray @ ego_vel_rig)
    measured = np.asarray(radial_velocity, dtype=float)
    residual = np.abs(measured - expected)
    if doppler_ambiguity is None:
        return residual
    # The true velocity is only known modulo the unambiguous interval, so try the
    # folds either side and keep the one that explains the return best. The
    # smallest consistent residual is the most that can honestly be claimed.
    interval = np.asarray(doppler_ambiguity, dtype=float)
    for k in range(-max_fold, max_fold + 1):
        if k == 0:
            continue
        residual = np.minimum(residual, np.abs(measured + k * interval - expected))
    return residual


def front_fov_mask(points_rig, fov_deg=60.0, max_range_m=300.0):
    """Keep only what a front-mounted sensor could see.

    Labels span the full 360 deg because they were produced from the complete
    rig, but this download has one forward camera and forward radars. Training
    against out-of-sector labels penalises the model for physics it cannot
    observe and teaches a language model to hallucinate objects behind the car.
    """
    x, y = points_rig[..., 0], points_rig[..., 1]
    azimuth = np.degrees(np.arctan2(y, x))
    ground_range = np.hypot(x, y)
    return (np.abs(azimuth) <= fov_deg) & (ground_range <= max_range_m)


def points_in_box(points, center, size, quaternion, margin=1.3):
    """Boolean mask of points inside an oriented 3D box."""
    rot = Rotation.from_quat(quaternion).as_matrix()
    local = (points - center) @ rot
    half = np.asarray(size, dtype=float) / 2.0 * margin
    return np.all(np.abs(local) <= half, axis=-1)


# --------------------------------------------------------------------------
# camera projection
# --------------------------------------------------------------------------
#
# The front wide camera is an f-theta (fisheye-family) 120 deg lens, not a
# pinhole, so `K @ [x, y, z]` does not apply. The calibration supplies the
# forward polynomial directly -- angle from the optical axis to pixel distance
# from the principal point -- which is all a projection needs; the inverse
# polynomial it also ships is for unprojection and is not used here.

def ftheta_project(points_cam, model):
    """Camera-frame points to pixels. NaN where a point is behind or outside.

    `points_cam` is (N, 3) with z along the optical axis. `model` is the parsed
    `model_parameters` JSON: `principal_point`, `angle_to_pixeldist_poly`,
    `max_angle`, `linear_cde`, `resolution`.

    Points beyond `max_angle` are dropped rather than clamped. The polynomial is
    only fitted inside the lens's field of view, and evaluating it outside
    returns a plausible-looking pixel for a direction the camera cannot see --
    which would put a box on an object that is not in the image.
    """
    xyz = np.asarray(points_cam, dtype=np.float64)
    if xyz.ndim == 1:
        xyz = xyz[None, :]
    radial = np.hypot(xyz[:, 0], xyz[:, 1])
    theta = np.arctan2(radial, xyz[:, 2])

    poly = np.asarray(model["angle_to_pixeldist_poly"], dtype=np.float64)
    # Stored lowest-order first; polyval wants the reverse.
    distance = np.polyval(poly[::-1], theta)

    with np.errstate(invalid="ignore", divide="ignore"):
        scale = np.where(radial > 1e-9, distance / radial, 0.0)
    x = xyz[:, 0] * scale
    y = xyz[:, 1] * scale

    c, d, e = model.get("linear_cde", [1.0, 0.0, 0.0])
    u = c * x + d * y
    v = e * x + y

    cx, cy = model["principal_point"]
    pixels = np.column_stack([u + cx, v + cy])
    outside = (theta > model["max_angle"]) | (xyz[:, 2] <= 0)
    pixels[outside] = np.nan
    return pixels


def box_corners(center, size, quaternion):
    """The eight corners of an oriented 3D box, in the box's parent frame."""
    l, w, h = size
    unit = np.array([[sx, sy, sz]
                     for sx in (-0.5, 0.5)
                     for sy in (-0.5, 0.5)
                     for sz in (-0.5, 0.5)], dtype=np.float64)
    local = unit * np.array([l, w, h], dtype=np.float64)
    rot = Rotation.from_quat(np.asarray(quaternion, dtype=np.float64))
    return rot.apply(local) + np.asarray(center, dtype=np.float64)


def rig_to_camera(points_rig, cam_rot, cam_translation):
    """Rig points into the camera frame.

    `cam_rot` is a rotation matrix, as `extrinsics_to_matrix` returns -- the
    same convention `sensor_to_rig` takes, so the two stay interchangeable.

    The extrinsics give the camera's pose *in* the rig, so a point needs the
    inverse transform: subtract the camera's position, then rotate by the
    transpose. `sensor_to_rig` is `p @ R.T + t`; this is its inverse.
    """
    shifted = np.asarray(points_rig, dtype=np.float64) - np.asarray(
        cam_translation, dtype=np.float64)
    return shifted @ np.asarray(cam_rot, dtype=np.float64)


def image_bbox(center, size, quaternion, cam_rot, cam_translation, model,
               min_visible=4):
    """Axis-aligned image box for a 3D box, or None if it is not visible.

    Built from the projected corners rather than from the centre and a scale:
    an f-theta lens bends straight lines, so a box's image extent is not a
    rectangle around its projected centre, and near the edge of the frame the
    difference is tens of pixels.

    `min_visible` guards against a box that is mostly behind the camera, where
    one or two corners project to a corner of the frame and imply a box that
    covers half the image.
    """
    corners = box_corners(center, size, quaternion)
    pixels = ftheta_project(rig_to_camera(corners, cam_rot, cam_translation),
                            model)
    good = pixels[~np.isnan(pixels).any(axis=1)]
    if len(good) < min_visible:
        return None
    width, height = model["resolution"]
    x1, y1 = good.min(axis=0)
    x2, y2 = good.max(axis=0)
    # Clip to the sensor, then reject anything that falls off it entirely.
    x1, x2 = np.clip([x1, x2], 0, width - 1)
    y1, y2 = np.clip([y1, y2], 0, height - 1)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return [float(x1), float(y1), float(x2), float(y2)]


def ftheta_unproject(pixel, model):
    """A pixel back to a unit ray in the camera frame.

    The calibration ships the inverse polynomial as well -- pixel distance from
    the principal point to angle from the optical axis -- so this is a lookup
    rather than a numerical inversion of the forward map.
    """
    cx, cy = model["principal_point"]
    c, d, e = model.get("linear_cde", [1.0, 0.0, 0.0])
    u, v = float(pixel[0]) - cx, float(pixel[1]) - cy
    det = c - d * e
    x = (u - d * v) / det
    y = (-e * u + c * v) / det
    radius = np.hypot(x, y)
    theta = np.polyval(
        np.asarray(model["pixeldist_to_angle_poly"], dtype=np.float64)[::-1],
        radius)
    if radius < 1e-9:
        return np.array([0.0, 0.0, 1.0])
    return np.array([x / radius * np.sin(theta), y / radius * np.sin(theta),
                     np.cos(theta)])


def camera_azimuth(pixel, model, cam_rot):
    """Bearing in the rig frame, in degrees, positive left.

    This is the one geometric quantity a single camera measures well: the
    direction to an object, from where it sits in the image. Measured on 2,717
    objects, the bearing recovered from a box centre sits a median 0.94 deg from
    the label, and within 5 deg 93.7% of the time. Range it cannot give at all,
    which is the half the radar supplies.
    """
    ray = np.asarray(cam_rot, dtype=np.float64) @ ftheta_unproject(pixel, model)
    return float(np.degrees(np.arctan2(ray[1], ray[0])))
