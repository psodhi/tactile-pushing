// Copyright (c) Facebook, Inc. and its affiliates.

/**
 * @file TactileRelativeTfPredictionFactor.cpp
 * @brief relative transform prediction network from tactile image feature pair inputs (vars: obj, ee poses)
 * @author Paloma Sodhi
 */

#ifndef TACTILE_RELATIVE_TF_PREDICTION_FACTOR
#define TACTILE_RELATIVE_TF_PREDICTION_FACTOR

#include <gtsam/base/Matrix.h>
#include <gtsam/base/Vector.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <cmath>
#include <iostream>
#include <string>

namespace pushestcpp {

class TactileRelativeTfPredictionFactor : public gtsam::NoiseModelFactor4<gtsam::Pose2, gtsam::Pose2, gtsam::Pose2, gtsam::Pose2> {
 private:
  bool useAnalyticJacobians_;
  gtsam::Vector imgFeat1_, imgFeat2_;

  bool yawOnlyError_;
  bool constantModel_;

  bool oracleFactor_;
  gtsam::Pose2 oraclePose_;

  int classLabel_;
  int numClasses_;

  std::shared_ptr<torch::jit::script::Module> torchModel_;

 public:
  TactileRelativeTfPredictionFactor(gtsam::Key objKey1, gtsam::Key objKey2, gtsam::Key eeKey1, gtsam::Key eeKey2,
                                    const gtsam::Vector& imgFeat1, const gtsam::Vector& imgFeat2, const std::string& torchModelFile,
                                    const gtsam::SharedNoiseModel& noiseModel = nullptr)
      : NoiseModelFactor4(noiseModel, objKey1, objKey2, eeKey1, eeKey2), imgFeat1_(imgFeat1), imgFeat2_(imgFeat2), useAnalyticJacobians_(false) {
    torch::jit::script::Module model = torch::jit::load(torchModelFile);
    torchModel_ = std::shared_ptr<torch::jit::script::Module>(new torch::jit::script::Module(model));

    yawOnlyError_ = false;
    constantModel_ = false;
    oracleFactor_ = false;
  }

  void setFlags(const bool yawOnlyError, const bool constantModel) {
    yawOnlyError_ = yawOnlyError;
    constantModel_ = constantModel;
  }

  void setOracle(const bool oracleFactor, const gtsam::Pose2 oraclePose) {
    oracleFactor_ = oracleFactor;
    oraclePose_ = oraclePose;
  }

  void setLabel(int classLabel, int numClasses) {
    classLabel_ = classLabel;
    numClasses_ = numClasses;
  }

  gtsam::Vector3 transformError(const gtsam::Pose2& objPose1, const gtsam::Pose2& objPose2, const gtsam::Pose2& eePose1, const gtsam::Pose2& eePose2, const gtsam::Pose2& poseRelMeas) const {
    // T_arg1^{-1} * T_arg2: relative pose between arg1, arg2 in arg1 coordinate frame
    gtsam::Pose2 eePose1__obj = objPose1.between(eePose1);
    gtsam::Pose2 eePose2__obj = objPose2.between(eePose2);
    gtsam::Pose2 poseRelExpect = eePose1__obj.between(eePose2__obj);
    gtsam::Pose2 errPose = poseRelExpect.between(poseRelMeas);

    if (yawOnlyError_) {
      gtsam::Pose2 poseRelExpectYaw = gtsam::Pose2(0, 0, poseRelExpect.theta()); 
      gtsam::Pose2 poseRelMeasYaw = gtsam::Pose2(0, 0, poseRelMeas.theta());
      errPose = poseRelExpectYaw.between(poseRelMeasYaw);
    }

    gtsam::Vector3 errVec = gtsam::Pose2::Logmap(errPose);

    return errVec;
  }

  gtsam::Vector evaluateError(const gtsam::Pose2& objPose1, const gtsam::Pose2& objPose2, const gtsam::Pose2& eePose1, const gtsam::Pose2& eePose2,
                              const boost::optional<gtsam::Matrix&> H1 = boost::none,
                              const boost::optional<gtsam::Matrix&> H2 = boost::none,
                              const boost::optional<gtsam::Matrix&> H3 = boost::none,
                              const boost::optional<gtsam::Matrix&> H4 = boost::none) const {
    // measurement: predicted relative pose from torch network model
    gtsam::Vector4 torchModelOutput = evaluateTorchModel();

    double yaw = -atan(torchModelOutput[3] / torchModelOutput[2]);

    if (yawOnlyError_) {
      yaw = asin(torchModelOutput[3]);
    }
    if (constantModel_) {
      yaw = 0.0;
    }

    gtsam::Pose2 poseRelMeas = gtsam::Pose2(torchModelOutput[0], torchModelOutput[1], yaw);

    if (oracleFactor_) {
      poseRelMeas = oraclePose_;
    }

    // error between measured and expected relative pose
    gtsam::Vector3 errVec = TactileRelativeTfPredictionFactor::transformError(objPose1, objPose2, eePose1, eePose2, poseRelMeas);

    gtsam::Matrix J1, J2, J3, J4;

    if (useAnalyticJacobians_) {
      // todo: add analytic derivative
    } else {
      J1 = gtsam::numericalDerivative11<gtsam::Vector3, gtsam::Pose2>(boost::bind(&TactileRelativeTfPredictionFactor::transformError, this, _1, objPose2, eePose1, eePose2, poseRelMeas), objPose1);
      J2 = gtsam::numericalDerivative11<gtsam::Vector3, gtsam::Pose2>(boost::bind(&TactileRelativeTfPredictionFactor::transformError, this, objPose1, _1, eePose1, eePose2, poseRelMeas), objPose2);
      J3 = gtsam::numericalDerivative11<gtsam::Vector3, gtsam::Pose2>(boost::bind(&TactileRelativeTfPredictionFactor::transformError, this, objPose1, objPose2, _1, eePose2, poseRelMeas), eePose1);
      J4 = gtsam::numericalDerivative11<gtsam::Vector3, gtsam::Pose2>(boost::bind(&TactileRelativeTfPredictionFactor::transformError, this, objPose1, objPose2, eePose1, _1, poseRelMeas), eePose2);
    }

    if (H1) *H1 = J1;
    if (H2) *H2 = J2;
    if (H3) *H3 = J3;
    if (H4) *H4 = J4;

    return errVec;
  }

  gtsam::Vector evaluateTorchModel(boost::optional<gtsam::Matrix&> H = boost::none) const {
    int modelOutputDim = 4;

    std::vector<torch::jit::IValue> inputI, inputJ;

    torch::Tensor imgFeatTensor1 = torch::tensor({imgFeat1_[0], imgFeat1_[1]});
    torch::Tensor imgFeatTensor2 = torch::tensor({imgFeat2_[0], imgFeat2_[1]});
    // torch::Tensor imgFeatTensor1 = torch::tensor({imgFeat1_[0], imgFeat1_[1], imgFeat1_[2], imgFeat1_[3], imgFeat1_[4], imgFeat1_[5]});
    // torch::Tensor imgFeatTensor2 = torch::tensor({imgFeat2_[0], imgFeat2_[1], imgFeat2_[2], imgFeat2_[3], imgFeat2_[4], imgFeat2_[5]});

    torch::Tensor classLabelVec = torch::nn::functional::one_hot(torch::tensor({classLabel_}), numClasses_);

    std::vector<torch::jit::IValue> input;
    input.push_back(imgFeatTensor1.unsqueeze(0));
    input.push_back(imgFeatTensor2.unsqueeze(0));
    input.push_back(classLabelVec.unsqueeze(0));

    at::Tensor outputTensor = torchModel_->forward(input).toTensor();

    gtsam::Vector outputVec = gtsam::Vector::Zero(modelOutputDim);
    for (size_t i = 0; i < modelOutputDim; i++) {
      outputVec[i] = outputTensor[0][i].item<double>();
    }

    return outputVec;
  }

  gtsam::Pose2 getMeasTransform() {
    gtsam::Vector4 torchModelOutput = evaluateTorchModel();

    double yaw = -atan(torchModelOutput[3] / torchModelOutput[2]);
    if (yawOnlyError_) {
      yaw = asin(torchModelOutput[3]);
    }
    if (constantModel_) {
      yaw = 0.0;
    }

    gtsam::Pose2 poseRelMeas = gtsam::Pose2(torchModelOutput[0], torchModelOutput[1], yaw);
  
    if (yawOnlyError_) {
      poseRelMeas = gtsam::Pose2(0.0, 0.0, yaw);
    }

    return poseRelMeas;
  }

  gtsam::Pose2 getExpectTransform(const gtsam::Pose2& objPose1, const gtsam::Pose2& objPose2, const gtsam::Pose2& eePose1, const gtsam::Pose2& eePose2) {
    // T_arg1^{-1} * T_arg2: relative pose between arg1, arg2 in arg1 coordinate frame
    gtsam::Pose2 eePose1__obj = objPose1.between(eePose1);
    gtsam::Pose2 eePose2__obj = objPose2.between(eePose2);
    gtsam::Pose2 poseRelExpect = eePose1__obj.between(eePose2__obj);

    if (yawOnlyError_) {
      poseRelExpect = gtsam::Pose2(0.0, 0.0, poseRelExpect.theta()); 
    }

    return poseRelExpect;
  }
};

}  // namespace pushestcpp
#endif