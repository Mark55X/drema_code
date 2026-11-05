// C entry points to GeoFIK for ctypes. GeoFIK reports every pose whose kinematic chain does not close on std::cout;
// it is muted during the call.
#include <array>
#include <iostream>
#include "geofik.h"

extern "C" {

// q7-locked IK for the Franka Hand frame: up to 8 solutions, rows of NaN for the missing ones.
int geofik_ik_q7(const double* r, const double* R_rowmajor, double q7, double q1_sing, double* out) {
    std::array<double, 3> position{r[0], r[1], r[2]};
    std::array<double, 9> rotation;
    for (int i = 0; i < 9; ++i) rotation[i] = R_rowmajor[i];
    std::array<std::array<double, 7>, 8> solutions;
    std::streambuf* console = std::cout.rdbuf(nullptr);
    unsigned int n = franka_ik_q7(position, rotation, q7, solutions, q1_sing);
    std::cout.rdbuf(console);
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 7; ++j) out[i * 7 + j] = solutions[i][j];
    return static_cast<int>(n);
}

// GeoFIK's own forward kinematics of the Franka Hand frame, used by the tests as an independent reference.
void geofik_fk(const double* q, double* T_rowmajor) {
    std::array<double, 7> joints;
    for (int i = 0; i < 7; ++i) joints[i] = q[i];
    Eigen::Matrix4d T = franka_fk(joints, 'E');
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) T_rowmajor[i * 4 + j] = T(i, j);
}
}
