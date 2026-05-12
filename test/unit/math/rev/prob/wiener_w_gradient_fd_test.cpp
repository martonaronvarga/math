#include <stan/math/rev.hpp>
#include <gtest/gtest.h>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct Case {
  const char* name;
  double y;
  double a;
  double t0;
  double w;
  double v;
  double sv;
  double sw;
  double st0;
};

struct Ad5Result {
  double lp;
  double gy;
  double ga;
  double gt0;
  double gw;
  double gv;
  double gsv;
};

struct AdFullResult {
  double lp;
  double gy;
  double ga;
  double gt0;
  double gw;
  double gv;
  double gsv;
  double gsw;
  double gst0;
};

template <typename F>
double central_diff(F&& f, double x, double h) {
  return (f(x + h) - f(x - h)) / (2.0 * h);
}

double lp5_double(const Case& c, double w) {
  return stan::math::wiener_lpdf(c.y, c.a, c.t0, w, c.v, c.sv);
}

double lp_full_double(const Case& c, double w) {
  return stan::math::wiener_lpdf(c.y, c.a, c.t0, w, c.v, c.sv, c.sw, c.st0);
}

Ad5Result ad5(const Case& c) {
  using stan::math::var;

  var y = c.y;
  var a = c.a;
  var t0 = c.t0;
  var w = c.w;
  var v = c.v;
  var sv = c.sv;

  var lp = stan::math::wiener_lpdf(y, a, t0, w, v, sv);
  lp.grad();

  Ad5Result out{lp.val(), y.adj(), a.adj(), t0.adj(), w.adj(), v.adj(),
                sv.adj()};

  stan::math::recover_memory();
  return out;
}

AdFullResult ad_full(const Case& c) {
  using stan::math::var;

  var y = c.y;
  var a = c.a;
  var t0 = c.t0;
  var w = c.w;
  var v = c.v;
  var sv = c.sv;
  var sw = c.sw;
  var st0 = c.st0;

  var lp = stan::math::wiener_lpdf(y, a, t0, w, v, sv, sw, st0);
  lp.grad();

  AdFullResult out{lp.val(), y.adj(),  a.adj(),  t0.adj(), w.adj(),
                   v.adj(),  sv.adj(), sw.adj(), st0.adj()};

  stan::math::recover_memory();
  return out;
}

std::string case_string(const Case& c) {
  std::ostringstream ss;
  ss << std::setprecision(17)
     << c.name << ": "
     << "y=" << c.y << ", "
     << "a=" << c.a << ", "
     << "t0=" << c.t0 << ", "
     << "w=" << c.w << ", "
     << "v=" << c.v << ", "
     << "sv=" << c.sv << ", "
     << "sw=" << c.sw << ", "
     << "st0=" << c.st0;
  return ss.str();
}

void print_w_fd_trace_5(const Case& c) {
  const Ad5Result ad = ad5(c);

  std::cout << std::setprecision(17);
  std::cout << "\nCASE_5 " << case_string(c) << "\n";
  std::cout << "AD_5 lp=" << ad.lp << " gw=" << ad.gw << "\n";

  for (double h : {1e-3, 1e-4, 1e-5, 1e-6, 1e-7}) {
    const double fd = central_diff([&](double ww) { return lp5_double(c, ww); },
                                   c.w, h);
    std::cout << "FD_5 h=" << h << " gw_fd=" << fd
              << " diff_ad_minus_fd=" << (ad.gw - fd) << "\n";
  }

  for (double ww : {c.w - 0.02, c.w - 0.01, c.w, c.w + 0.01, c.w + 0.02}) {
    if (ww > 0.0 && ww < 1.0) {
      std::cout << "GRID_5 w=" << ww << " lp=" << lp5_double(c, ww) << "\n";
    }
  }
}

void print_w_fd_trace_full(const Case& c) {
  const AdFullResult ad = ad_full(c);

  std::cout << std::setprecision(17);
  std::cout << "\nCASE_FULL " << case_string(c) << "\n";
  std::cout << "AD_FULL lp=" << ad.lp << " gw=" << ad.gw << "\n";

  for (double h : {1e-3, 1e-4, 1e-5, 1e-6, 1e-7}) {
    const double fd = central_diff(
        [&](double ww) { return lp_full_double(c, ww); }, c.w, h);
    std::cout << "FD_FULL h=" << h << " gw_fd=" << fd
              << " diff_ad_minus_fd=" << (ad.gw - fd) << "\n";
  }

  for (double ww : {c.w - 0.02, c.w - 0.01, c.w, c.w + 0.01, c.w + 0.02}) {
    if (ww > 0.0 && ww < 1.0
        && (c.sw == 0.0 || (ww - c.sw / 2.0 > 0.0 && ww + c.sw / 2.0 < 1.0))) {
      std::cout << "GRID_FULL w=" << ww << " lp=" << lp_full_double(c, ww)
                << "\n";
    }
  }
}

double endpoint_identity_full_sw_only(const Case& c) {
  // Valid when sw > 0 and st0 == 0.
  //
  // L(w0) = log((1 / sw) int_{w0-sw/2}^{w0+sw/2} f(u) du)
  //
  // dL/dw0 = [f(w0 + sw/2) - f(w0 - sw/2)] / [sw * F],
  // where F is the marginal density exp(L).
  const double low = c.w - c.sw / 2.0;
  const double high = c.w + c.sw / 2.0;

  Case low_case = c;
  Case high_case = c;
  Case mid_case = c;

  low_case.sw = 0.0;
  low_case.st0 = 0.0;
  low_case.w = low;

  high_case.sw = 0.0;
  high_case.st0 = 0.0;
  high_case.w = high;

  const double f_low = std::exp(lp5_double(low_case, low));
  const double f_high = std::exp(lp5_double(high_case, high));
  const double marginal_density = std::exp(lp_full_double(mid_case, c.w));

  return (f_high - f_low) / (c.sw * marginal_density);
}

std::vector<Case> stan_existing_rows() {
  return {
      {"row_0", 2.0, 2.0, 1e-9, 0.10, 2.0, 0.0, 0.00, 0.000},
      {"row_1", 3.0, 2.0, 0.01, 0.50, 2.0, 0.2, 0.00, 0.000},
      {"row_2", 4.0, 10.0, 0.01, 0.80, 4.0, 0.0, 0.10, 0.000},
      {"row_3", 5.0, 4.0, 0.01, 0.70, 3.0, 0.0, 0.00, 0.007},
      {"row_4", 6.0, 10.0, 0.01, 0.10, -3.0, 0.2, 0.10, 0.000},
      {"row_5", 7.0, 1.0, 0.01, 0.90, 1.0, 0.2, 0.00, 0.007},
      {"row_6", 8.0, 3.0, 0.01, 0.70, -1.0, 0.0, 0.10, 0.007},
      {"row_7", 8.85, 1.7, 0.01, 0.92, -7.3, 0.7, 0.01, 0.009},
      {"row_8", 8.9, 2.4, 0.01, 0.90, -4.9, 0.0, 0.00, 0.009},
      {"row_9", 9.0, 11.0, 0.01, 0.12, 4.5, 0.7, 0.10, 0.009},
      {"row_10", 1.0, 1.5, 0.10, 0.50, 3.0, 0.5, 0.20, 0.000},
  };
}

}  // namespace

TEST(MathRevProbWienerWGradientFD, FiveParamI4AdDisagreesWithOwnValueFD) {
  const Case c{"five_param_i4", 6.0, 10.0, 0.01, 0.10, -3.0, 0.2,
               0.0, 0.0};

  print_w_fd_trace_5(c);

  const Ad5Result ad = ad5(c);
  const double fd = central_diff([&](double ww) { return lp5_double(c, ww); },
                                 c.w, 1e-6);

  // This is the assertion that should currently fail if the bug is present.
  EXPECT_NEAR(ad.gw, fd, 1e-5)
      << "Stan reverse-mode gw disagrees with central finite difference of "
         "Stan Math's own wiener_lpdf(y,a,t0,w,v,sv). "
      << case_string(c);
}

TEST(MathRevProbWienerWGradientFD, FiveParamZeroSvControlAlsoFails) {
  const Case c{"five_param_i4_sv_zero", 6.0, 10.0, 0.01, 0.10, -3.0,
               0.0, 0.0, 0.0};

  print_w_fd_trace_5(c);

  const Ad5Result ad = ad5(c);
  const double fd = central_diff([&](double ww) { return lp5_double(c, ww); },
                                 c.w, 1e-6);

  // This control distinguishes "sv derivative/prefactor bug" from a more
  // basic w-adjoint bug.
  EXPECT_NEAR(ad.gw, fd, 1e-5)
      << "The w-adjoint mismatch persists even at sv = 0. "
      << case_string(c);
}

TEST(MathRevProbWienerWGradientFD, FullI4AdDisagreesWithOwnValueFD) {
  const Case c{"full_i4_sw_positive", 6.0, 10.0, 0.01, 0.10, -3.0, 0.2,
               0.1, 0.0};

  print_w_fd_trace_full(c);

  const AdFullResult ad = ad_full(c);
  const double fd = central_diff(
      [&](double ww) { return lp_full_double(c, ww); }, c.w, 1e-6);

  // This is the propagated full-parameter failure.
  EXPECT_NEAR(ad.gw, fd, 1e-5)
      << "Stan reverse-mode gw disagrees with central finite difference of "
         "Stan Math's own full wiener_lpdf value function. "
      << case_string(c);
}

TEST(MathRevProbWienerWGradientFD, FullI4EndpointIdentity) {
  const Case c{"full_i4_endpoint_identity", 6.0, 10.0, 0.01, 0.10, -3.0,
               0.2, 0.1, 0.0};

  ASSERT_GT(c.sw, 0.0);
  ASSERT_EQ(c.st0, 0.0);

  const AdFullResult ad = ad_full(c);
  const double fd = central_diff(
      [&](double ww) { return lp_full_double(c, ww); }, c.w, 1e-6);
  const double endpoint = endpoint_identity_full_sw_only(c);

  std::cout << std::setprecision(17)
            << "\nENDPOINT_IDENTITY " << case_string(c) << "\n"
            << "AD_FULL gw=" << ad.gw << "\n"
            << "FD_FULL gw=" << fd << "\n"
            << "ENDPOINT gw=" << endpoint << "\n";

  // This checks that the independent endpoint identity agrees with finite
  // differences of the value function.
  EXPECT_NEAR(endpoint, fd, 1e-5);

  // This is expected to fail if the current Stan adjoint is wrong.
  EXPECT_NEAR(ad.gw, endpoint, 1e-5)
      << "For sw > 0 and st0 == 0, d/dw has a closed endpoint identity. "
         "Stan AD disagrees with both that identity and finite differences. "
      << case_string(c);
}

TEST(MathRevProbWienerWGradientFD, ExistingFullRowsCompareWAdjointToFD) {
  const auto rows = stan_existing_rows();

  for (const auto& c : rows) {
    SCOPED_TRACE(case_string(c));

    // Avoid cases where finite differencing w would violate Stan's support
    // constraint w - sw/2 > 0 and w + sw/2 < 1.
    const double h = 1e-6;
    if (!(c.w - h > 0.0 && c.w + h < 1.0)) {
      continue;
    }
    if (c.sw > 0.0
        && !(c.w - h - c.sw / 2.0 > 0.0
             && c.w + h + c.sw / 2.0 < 1.0)) {
      continue;
    }

    const AdFullResult ad = ad_full(c);
    const double fd = central_diff(
        [&](double ww) { return lp_full_double(c, ww); }, c.w, h);

    std::cout << std::setprecision(17)
              << "ROW_CHECK " << case_string(c)
              << " lp=" << ad.lp
              << " ad_gw=" << ad.gw
              << " fd_gw=" << fd
              << " diff=" << (ad.gw - fd)
              << "\n";

    EXPECT_NEAR(ad.gw, fd, 1e-4)
        << "w-adjoint does not match central finite difference for existing "
           "Wiener test row.";
  }
}

TEST(MathRevProbWienerWGradientFD, FiveAndFullZeroValueAndWAdjointSamePath) {
  const Case c{"five_vs_full_zero", 6.0, 10.0, 0.01, 0.10, -3.0, 0.2,
               0.0, 0.0};

  const Ad5Result ad5_result = ad5(c);
  const AdFullResult ad_full_result = ad_full(c);

  std::cout << std::setprecision(17)
            << "\nFIVE_VS_FULL_ZERO " << case_string(c) << "\n"
            << "AD_5 lp=" << ad5_result.lp
            << " gw=" << ad5_result.gw << "\n"
            << "AD_FULL_ZERO lp=" << ad_full_result.lp
            << " gw=" << ad_full_result.gw << "\n";

  EXPECT_NEAR(ad5_result.lp, ad_full_result.lp, 1e-12);
  EXPECT_NEAR(ad5_result.gw, ad_full_result.gw, 1e-12);
}
