// Copyright (c) Facebook, Inc. and its affiliates.

/**
 * @file IntersectionPlanarSDFObjFactor.h
 * @brief Intersection factor using 2D signed distance field (vars: obj poses)
 * @author Paloma Sodhi
 */

#ifndef INTERSECTION_PLANAR_SDF_OBJ_FACTOR_H_
#define INTERSECTION_PLANAR_SDF_OBJ_FACTOR_H_

#include <gtsam/base/Matrix.h>
#include <gtsam/base/Vector.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <pushestcpp/thirdparty/gpmp2/PlanarSDF.h>

namespace pushestcpp {

class IntersectionPlanarSDFObjFactor : public gtsam::NoiseModelFactor1<gtsam::Pose2> {
 private:
  PlanarSDF sdf_;
  gtsam::Vector2 eeCenter_;
  double eeRadius_;

  bool useAnalyticJacobians_;

 public:
  IntersectionPlanarSDFObjFactor(gtsam::Key key, const PlanarSDF& sdf, const gtsam::Vector2& eeCenter, const double& eeRadius, const gtsam::SharedNoiseModel& model = nullptr)
      : NoiseModelFactor1(model, key), sdf_(sdf), eeCenter_(eeCenter), eeRadius_(eeRadius), useAnalyticJacobians_(false) {}

  gtsam::Vector1 IntersectionErrorOneSidedHinge(const gtsam::Pose2& objPose) const {
    double dist, err;
    gtsam::Vector1 errVec;
    gtsam::Point2 eeCenter__obj = objPose.transformTo(gtsam::Point2(eeCenter_));

    try {
      dist = sdf_.getSignedDistance(eeCenter__obj);
    } catch (SDFQueryOutOfRange&) {
      std::cout << "[IntersectionPlanarSDFFactor] WARNING: SDF query pos (" << eeCenter__obj.x()
                << ", " << eeCenter__obj.y() << ") out of range. Setting error to 0. " << std::endl;
      errVec << 0.0;
      return errVec;
    }

    if (dist > eeRadius_) {  // not penetrating
      err = 0;
    } else {
      err = eeRadius_ - dist;
    }
    errVec << err;

    return errVec;
  }

  gtsam::Vector evaluateError(const gtsam::Pose2& p, boost::optional<gtsam::Matrix&> H = boost::none) const {
    gtsam::Vector errVec = IntersectionErrorOneSidedHinge(p);

    gtsam::Matrix J;
    if (useAnalyticJacobians_) {
      // todo: add analytic derivative
    } else {
      J = gtsam::numericalDerivative11<gtsam::Vector1, gtsam::Pose2>(boost::bind(&IntersectionPlanarSDFObjFactor::IntersectionErrorOneSidedHinge, this, _1), p);
    }

    if (H) *H = J;

    return errVec;
  }
};

}  // namespace pushestcpp

#endif