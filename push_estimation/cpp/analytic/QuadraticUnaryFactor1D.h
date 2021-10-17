#ifndef QUADRATIC_UNARY_FACTOR_1D_H_
#define QUADRATIC_UNARY_FACTOR_1D_H_

#include <math.h>

#include <gtsam/base/Matrix.h>
#include <gtsam/base/Vector.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

namespace pushest {

class QuadraticUnaryFactor1D : public gtsam::NoiseModelFactor1<gtsam::Vector> {
 private:
  gtsam::Vector meas_;

 public:
   QuadraticUnaryFactor1D(gtsam::Key varKey, const gtsam::Vector1 &meas, const int constraintFactorType, const gtsam::SharedNoiseModel &model = nullptr)
       : NoiseModelFactor1(model, varKey), meas_(meas)
   {}

  gtsam::Vector evaluateError(const gtsam::Vector& x, boost::optional<gtsam::Matrix&> H = boost::none) const {
    
    // compute jacobian
    gtsam::Matrix jac(1,1);
    jac << 1.;
    if (H) *H = jac;

    // compute error
    gtsam::Vector errorVector(1);
    errorVector << (x - meas_);

    return errorVector;
  }

  gtsam::Vector constraintError(const gtsam::Value& val) const {
    gtsam::Vector x = val.cast<gtsam::Vector>();
    gtsam::Vector errorVector(1);
    errorVector <<  (x - meas_);

    return errorVector;
  }

  gtsam::Matrix constraintJacobian(const gtsam::Value& val) const {
    
    gtsam::Vector x = val.cast<gtsam::Vector>();

    gtsam::Matrix jac(1,1);
    jac << 1.;
    gtsam::Matrix G = jac;

    return G;
  }

};

}  // namespace pushest

#endif