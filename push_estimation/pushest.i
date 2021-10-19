// gtsam variables

class gtsam::Point2;
class gtsam::Pose2;
class gtsam::Vector3;

class gtsam::Point3;
class gtsam::Pose3;
class gtsam::Vector6;

class gtsam::Values;
virtual class gtsam::noiseModel::Base;
virtual class gtsam::NonlinearFactor;
virtual class gtsam::NonlinearFactorGraph;
virtual class gtsam::NoiseModelFactor : gtsam::NonlinearFactor;

namespace pushest {

#include <cpp/analytic/QuadraticUnaryFactor1D.h>
class QuadraticUnaryFactor1D: gtsam::NoiseModelFactor {
    QuadraticUnaryFactor1D(gtsam::Key varKey, const gtsam::Vector1 &meas, const int constraintFactorType, const gtsam::SharedNoiseModel &model = nullptr);
    gtsam::Vector evaluateError(const gtsam::Vector& x);
};

#include <cpp/analytic/QuadraticBinaryFactor1D.h>
class QuadraticBinaryFactor1D: gtsam::NoiseModelFactor {
  QuadraticBinaryFactor1D(gtsam::Key varKey1, gtsam::Key varKey2, const gtsam::Vector1 &meas, const gtsam::SharedNoiseModel &model = nullptr);
  gtsam::Vector evaluateError(const gtsam::Vector &x1, const gtsam::Vector &x2);
};

// #include <cpp/thirdparty/gpmp2/PlanarSDF.h>
// class PlanarSDF {
//   PlanarSDF(const gtsam::Point2& origin, double cell_size, const Matrix& data);
//   // access
//   double getSignedDistance(const gtsam::Point2& point) const;
//   void print(string s) const;
// };

// #include <cpp/dynamics/QSVelPushMotionRealObjEEFactor.h>
// virtual class QSVelPushMotionRealObjEEFactor : gtsam::NoiseModelFactor {
//   QSVelPushMotionRealObjEEFactor(size_t objKey0, size_t objKey1, size_t eeKey0, size_t eeKey1,
//                                  const double& cSq, const gtsam::noiseModel::Base* noiseModel);
//   Vector evaluateError(const gtsam::Pose2& objPose0, const gtsam::Pose2& objPose1,
//                        const gtsam::Pose2& eePose0, const gtsam::Pose2& eePose1) const;
// };

// #include <cpp/geometry/IntersectionPlanarSDFObjEEFactor.h>
// virtual class IntersectionPlanarSDFObjEEFactor : gtsam::NoiseModelFactor {
//   IntersectionPlanarSDFObjEEFactor(size_t objKey, size_t eeKey, const PlanarSDF& sdf, const double& eeRadius, const gtsam::noiseModel::Base* noiseModel);
//   Vector evaluateError(const gtsam::Pose2& objPose, const gtsam::Pose2& eePose) const;
// };

// #include <cpp/contact/TactileRelativeTfPredictionFactor.h>
// virtual class TactileRelativeTfPredictionFactor : gtsam::NoiseModelFactor {
//   TactileRelativeTfPredictionFactor(size_t objKey1, size_t objKey2, size_t eeKey1, size_t eeKey2,
//                                     const Vector& imgFeat1, const Vector& imgFeat2, string torchModelFile,
//                                     const gtsam::noiseModel::Base* noiseModel);
//   Vector evaluateError(const gtsam::Pose2& objPose1, const gtsam::Pose2& objPose2,
//                        const gtsam::Pose2& eePose1, const gtsam::Pose2& eePose2) const;
//   void setFlags(const bool yawOnlyError,const bool constantModel);
//   void setOracle(const bool oracleFactor, const gtsam::Pose2 oraclePose);
//   void setLabel(int classLabel, int numClasses);
//   gtsam::Pose2 getMeasTransform();
//   gtsam::Pose2 getExpectTransform(const gtsam::Pose2& objPose1, const gtsam::Pose2& objPose2,
//                                   const gtsam::Pose2& eePose1, const gtsam::Pose2& eePose2);
// };

}  // namespace pushest
