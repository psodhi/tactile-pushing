// Copyright (c) Facebook, Inc. and its affiliates.

/**
 * @file QSVelPushMotionFactor.h
 * @brief velocity only quasi-static pushing model factor (vars: obj poses)
 * @author Paloma Sodhi
 */

#ifndef QS_VEL_PUSH_MOTION_FACTOR_H_
#define QS_VEL_PUSH_MOTION_FACTOR_H_

#include <gtsam/base/Matrix.h>
#include <gtsam/base/Vector.h>
#include <gtsam/base/numericalDerivative.h>

#include <gtsam/geometry/Pose2.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

namespace pushestcpp {

class QSVelPushMotionMeasurementObj {
 public:
  gtsam::Vector2 contactPointPrev_;
  gtsam::Vector2 contactPointCurr_;

  double fMaxSq_, tauMaxSq_;

  QSVelPushMotionMeasurementObj(const gtsam::Vector2 contactPointPrev, const gtsam::Vector2 contactPointCurr, const double fMaxSq, const double tauMaxSq)
      : contactPointPrev_(contactPointPrev), contactPointCurr_(contactPointCurr), fMaxSq_(fMaxSq), tauMaxSq_(tauMaxSq) {}
};

class QSVelPushMotionObjFactor : public gtsam::NoiseModelFactor2<gtsam::Pose2, gtsam::Pose2> {
 private:
  QSVelPushMotionMeasurementObj pushMeas_;
  bool useAnalyticJacobians_;

 public:
  QSVelPushMotionObjFactor(gtsam::Key key1, gtsam::Key key2, const QSVelPushMotionMeasurementObj pushMeas, const gtsam::SharedNoiseModel& model = nullptr)
      : NoiseModelFactor2(model, key1, key2), pushMeas_(pushMeas), useAnalyticJacobians_(false) {}

  gtsam::Vector3 pushVelocityError(const gtsam::Pose2& p1, const gtsam::Pose2& p2) const {
    // p1: prev pose, p2: curr pose

    gtsam::Pose2 p2Ori = gtsam::Pose2(0, 0, p2.theta());
    gtsam::Pose2 poseBetween__world = p1.between(p2);

    // compute object velocities
    gtsam::Vector2 velXYObj__world = gtsam::Vector2(p2.x(), p2.y()) - gtsam::Vector2(p1.x(), p1.y());
    gtsam::Vector2 velXYObj__obj = p2Ori.transformTo(gtsam::Point2(velXYObj__world[0], velXYObj__world[1]));  // rotateTo()

    // compute contact point velocities
    gtsam::Vector2 velXYContact__world = pushMeas_.contactPointCurr_ - pushMeas_.contactPointPrev_;
    gtsam::Vector2 velXYContact__obj = p2Ori.transformTo(gtsam::Point2(velXYContact__world[0], velXYContact__world[1]));  // rotateTo()

    gtsam::Vector contactPoint__obj = p2.transformTo(gtsam::Point2(pushMeas_.contactPointCurr_[0], pushMeas_.contactPointCurr_[1]));

    double v_x = velXYObj__obj[0];
    double v_y = velXYObj__obj[1];
    double omega = poseBetween__world.theta();

    double v_px = velXYContact__obj[0];
    double v_py = velXYContact__obj[1];

    double px = contactPoint__obj[0];
    double py = contactPoint__obj[1];

    // D*V = Vp (Ref: Zhou '17)
    gtsam::Matrix33 D;
    gtsam::Vector3 V, Vp;

    D << 1, 0, -py, 0, 1, px, -pushMeas_.fMaxSq_ * py, pushMeas_.fMaxSq_ * px, -pushMeas_.tauMaxSq_;
    V << v_x, v_y, omega;
    Vp << v_px, v_py, 0;

    gtsam::Vector3 errVec;
    errVec = D * V - Vp;

    return errVec;
  }

  gtsam::Vector evaluateError(const gtsam::Pose2& p1, const gtsam::Pose2& p2, const boost::optional<gtsam::Matrix&> H1 = boost::none,
                       const boost::optional<gtsam::Matrix&> H2 = boost::none) const {
    gtsam::Vector3 errVec = QSVelPushMotionObjFactor::pushVelocityError(p1, p2);

    gtsam::Matrix J1, J2;

    if (useAnalyticJacobians_) {
      // todo: add analytic derivative
    } else {
      J1 = gtsam::numericalDerivative11<gtsam::Vector3, gtsam::Pose2>(boost::bind(&QSVelPushMotionObjFactor::pushVelocityError, this, _1, p2), p1);
      J2 = gtsam::numericalDerivative11<gtsam::Vector3, gtsam::Pose2>(boost::bind(&QSVelPushMotionObjFactor::pushVelocityError, this, p1, _1), p2);
    }

    if (H1) *H1 = J1;
    if (H2) *H2 = J2;

    return errVec;
  }
};

}  // namespace pushestcpp

#endif