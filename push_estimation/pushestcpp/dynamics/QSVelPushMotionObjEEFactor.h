// Copyright (c) Facebook, Inc. and its affiliates.

/**
 * @file QSVelPushMotionObjEEFactor.h
 * @brief velocity only quasi-static pushing model factor (vars: obj, ee poses)
 * @author Paloma Sodhi
 */

#ifndef QS_VEL_PUSH_MOTION_OBJ_EE_FACTOR_H_
#define QS_VEL_PUSH_MOTION_OBJ_EE_FACTOR_H_

#include <gtsam/base/Matrix.h>
#include <gtsam/base/Vector.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

namespace pushestcpp {

class QSVelPushMotionMeasurementObjEE {
 public:
  gtsam::Vector2 contactNormal0_;
  gtsam::Vector2 contactNormal1_;

  double eeRadius_;
  double fMaxSq_, tauMaxSq_;

  QSVelPushMotionMeasurementObjEE(const gtsam::Vector2& contactNormal0, const gtsam::Vector2& contactNormal1,
                                  const double& eeRadius, const double& fMaxSq, const double& tauMaxSq)
      : contactNormal0_(contactNormal0), contactNormal1_(contactNormal1), eeRadius_(eeRadius), fMaxSq_(fMaxSq), tauMaxSq_(tauMaxSq) {}
};

class QSVelPushMotionObjEEFactor : public gtsam::NoiseModelFactor4<gtsam::Pose2, gtsam::Pose2, gtsam::Pose2, gtsam::Pose2> {
 private:
  QSVelPushMotionMeasurementObjEE pushMeas_;
  bool useAnalyticJacobians_;

 public:
  QSVelPushMotionObjEEFactor(gtsam::Key objKey0, gtsam::Key objKey1, gtsam::Key eeKey0, gtsam::Key eeKey1,
                             const QSVelPushMotionMeasurementObjEE& pushMeas, const gtsam::SharedNoiseModel& model = nullptr)
      : NoiseModelFactor4(model, objKey0, objKey1, eeKey0, eeKey1), pushMeas_(pushMeas), useAnalyticJacobians_(false) {}

  gtsam::Vector3 pushVelocityError(const gtsam::Pose2& objPose0, const gtsam::Pose2& objPose1, const gtsam::Pose2& eePose0, const gtsam::Pose2& eePose1) const {
    gtsam::Pose2 objOri1 = gtsam::Pose2(0, 0, objPose1.theta());
    gtsam::Pose2 poseBetween__world = objPose0.between(objPose1);

    // compute object velocities using prev, curr obj poses
    gtsam::Vector2 velXYObj__world = gtsam::Vector2(objPose1.x(), objPose1.y()) - gtsam::Vector2(objPose0.x(), objPose0.y());
    gtsam::Vector2 velXYObj__obj = objOri1.transformTo(gtsam::Point2(velXYObj__world[0], velXYObj__world[1]));  // rotateTo()

    // compute contact point velocities using prev, curr endeff poses
    gtsam::Vector2 contactPoint0, contactPoint1;
    contactPoint0[0] = eePose0.x() + pushMeas_.eeRadius_ * pushMeas_.contactNormal0_[0];
    contactPoint0[1] = eePose0.y() + pushMeas_.eeRadius_ * pushMeas_.contactNormal0_[1];
    contactPoint1[0] = eePose1.x() + pushMeas_.eeRadius_ * pushMeas_.contactNormal1_[0];
    contactPoint1[1] = eePose1.y() + pushMeas_.eeRadius_ * pushMeas_.contactNormal1_[1];
    gtsam::Vector2 velXYContact__world = contactPoint1 - contactPoint0;
    gtsam::Vector2 velXYContact__obj = objOri1.transformTo(gtsam::Point2(velXYContact__world[0], velXYContact__world[1]));  // rotateTo()

    // current contact point in object frame
    gtsam::Vector contactPoint__obj = objPose1.transformTo(gtsam::Point2(contactPoint1[0], contactPoint1[1]));

    double vX = velXYObj__obj[0];
    double vY = velXYObj__obj[1];
    double omega = poseBetween__world.theta();

    double vPX = velXYContact__obj[0];
    double vPY = velXYContact__obj[1];

    double px = contactPoint__obj[0];
    double py = contactPoint__obj[1];

    // D*V = Vp (Ref: Zhou '17)
    gtsam::Matrix33 D;
    gtsam::Vector3 V, Vp;

    D << 1, 0, -py, 0, 1, px, -pushMeas_.fMaxSq_ * py, pushMeas_.fMaxSq_ * px, -pushMeas_.tauMaxSq_;
    V << vX, vY, omega;
    Vp << vPX, vPY, 0;

    gtsam::Vector3 errVec;
    errVec = D * V - Vp;
    
    return errVec;
  }

  // Matrix pushVelocityErrorJacobian(const Pose2& objPose0, const Pose2& objPose1, const Pose2& eePose0, const Pose2& eePose1, int idx) const {
  // }

  gtsam::Vector evaluateError(const gtsam::Pose2& objPose0, const gtsam::Pose2& objPose1, const gtsam::Pose2& eePose0, const gtsam::Pose2& eePose1,
                       const boost::optional<gtsam::Matrix&> H1 = boost::none,
                       const boost::optional<gtsam::Matrix&> H2 = boost::none,
                       const boost::optional<gtsam::Matrix&> H3 = boost::none,
                       const boost::optional<gtsam::Matrix&> H4 = boost::none) const {
    gtsam::Vector3 errVec = QSVelPushMotionObjEEFactor::pushVelocityError(objPose0, objPose1, eePose0, eePose1);

    gtsam::Matrix J1, J2, J3, J4;

    if (useAnalyticJacobians_) {
      // todo: add analytic derivative
      // J1 = QSVelPushMotionObjEEFactor::pushVelocityErrorJacobian(objPose0, objPose1, eePose0, eePose1, 1);
    } else {
      J1 = gtsam::numericalDerivative11<gtsam::Vector3, gtsam::Pose2>(boost::bind(&QSVelPushMotionObjEEFactor::pushVelocityError, this, _1, objPose1, eePose0, eePose1), objPose0);
      J2 = gtsam::numericalDerivative11<gtsam::Vector3, gtsam::Pose2>(boost::bind(&QSVelPushMotionObjEEFactor::pushVelocityError, this, objPose0, _1, eePose0, eePose1), objPose1);
      J3 = gtsam::numericalDerivative11<gtsam::Vector3, gtsam::Pose2>(boost::bind(&QSVelPushMotionObjEEFactor::pushVelocityError, this, objPose0, objPose1, _1, eePose1), eePose0);
      J4 = gtsam::numericalDerivative11<gtsam::Vector3, gtsam::Pose2>(boost::bind(&QSVelPushMotionObjEEFactor::pushVelocityError, this, objPose0, objPose1, eePose0, _1), eePose1);
    }

    if (H1) *H1 = J1;
    if (H2) *H2 = J2;
    if (H3) *H3 = J3;
    if (H4) *H4 = J4;

    return errVec;
  }
};

}  // namespace pushestcpp

#endif