#ifndef QUADRATIC_BINARY_FACTOR_1D_H_
#define QUADRATIC_BINARY_FACTOR_1D_H_

#include <math.h>

#include <gtsam/base/Matrix.h>
#include <gtsam/base/Vector.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

namespace pushest {

class QuadraticBinaryFactor1D : public gtsam::NoiseModelFactor2<gtsam::Vector, gtsam::Vector> {
 private:
  gtsam::Vector meas_;

 public:
   QuadraticBinaryFactor1D(gtsam::Key varKey1, gtsam::Key varKey2, const gtsam::Vector1 &meas, const gtsam::SharedNoiseModel &model = nullptr)
       : NoiseModelFactor2(model, varKey1, varKey2), meas_(meas)
   {}

   gtsam::Vector evaluateError(const gtsam::Vector &x1, const gtsam::Vector &x2, boost::optional<gtsam::Matrix &> H1 = boost::none, boost::optional<gtsam::Matrix &> H2 = boost::none) const
   {
       // compute jacobian
       if (H1) *H1 = (gtsam::Matrix11() << -1.).finished();
       if (H2) *H2 = (gtsam::Matrix11() << 1.).finished();

    // compute error
    gtsam::Vector errorVector(1);
    errorVector << ((x2 - x1) - meas_);
    
    return errorVector;
   }
};

}  // namespace pushest

#endif