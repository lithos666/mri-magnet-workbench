"""Analytic harmonic fields, angular teeth, and synthesis-axis regressions."""
import copy
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from mri_magnet.de_novo import _axis_features, _axis_order_weights, build_basis
from mri_magnet.geometry import Design, RingSpec
from mri_magnet.harmonics import (analyze_harmonics, azimuthal_fourier,
    fit_spherical_bz, plot_harmonics, sphere_quadrature)
from mri_magnet.sampling import sphere_samples
from mri_magnet.workbench_config import default_config


def fields(bz):
    return np.c_[np.zeros(len(bz)), np.zeros(len(bz)), bz]


class ComponentHarmonicsTests(unittest.TestCase):
    def setUp(self):
        self.points, self.weights, _ = sphere_quadrature(50., 8)

    def test_uniform_vector_field_has_no_nonconstant_component(self):
        vector = np.tile([.002, -.001, .04], (len(self.points), 1))
        report = fit_spherical_bz(self.points, vector, 50., self.weights)
        self.assertLess(report['residual_rms_ppm'], 1e-8)
        self.assertLess(max(row['rms_ppm'] for row in report['degree_rms_ppm']), 1e-8)
        self.assertAlmostEqual(report['reference_Bz_T'], .04, places=14)

    def test_known_z2_and_transverse_quadrupole_recover_in_separate_terms(self):
        x, y, z = self.points.T/50.
        # Both (3z²-r²)/2 and x²-y² are exact degree-two harmonic polynomials.
        bz = .04*(1 + 20e-6*(3*z*z-1)/2 + 12e-6*(x*x-y*y))
        report = fit_spherical_bz(self.points, fields(bz), 50., self.weights)
        self.assertAlmostEqual(report['zonal_ppm']['2'], 20., places=7)
        term = next(row for row in report['coefficients']
                    if (row['degree'], row['order'], row['component']) == (2, 2, 'cos'))
        self.assertAlmostEqual(term['coefficient_ppm'], 12./(math.sqrt(15.)/2), places=7)
        self.assertLess(report['residual_rms_ppm'], 1e-8)

    def test_rotation_moves_cosine_into_sine_without_changing_spectral_power(self):
        x, y, z = self.points.T/50.
        a = fit_spherical_bz(self.points, fields(.04*(1+12e-6*(x*x-y*y))), 50., self.weights)
        b = fit_spherical_bz(self.points, fields(.04*(1+24e-6*x*y)), 50., self.weights)
        self.assertAlmostEqual(a['degree_rms_ppm'][1]['rms_ppm'], b['degree_rms_ppm'][1]['rms_ppm'], places=8)
        term = next(row for row in b['coefficients']
                    if (row['degree'], row['order'], row['component']) == (2, 2, 'sin'))
        self.assertAlmostEqual(term['coefficient_ppm'], 12./(math.sqrt(15.)/2), places=7)

    def test_z8_is_resolved_and_omitted_degree_is_visible_in_residual(self):
        u = self.points[:, 2]/50.
        # Standard Legendre P8, evaluated independently by numpy.polynomial.
        series = np.zeros(9); series[8] = 1.
        bz = .04*(1+5e-6*np.polynomial.legendre.legval(u, series))
        full = fit_spherical_bz(self.points, fields(bz), 50., self.weights, 8)
        truncated = fit_spherical_bz(self.points, fields(bz), 50., self.weights, 4)
        self.assertAlmostEqual(full['zonal_ppm']['8'], 5., places=7)
        self.assertLess(full['residual_rms_ppm'], 1e-8)
        self.assertAlmostEqual(truncated['residual_rms_ppm'], 5/math.sqrt(17), places=7)

    def test_zero_reference_wrong_surface_and_unresolved_samples_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Reference Bz'):
            fit_spherical_bz(self.points, fields(np.zeros(len(self.points))), 50., self.weights)
        with self.assertRaisesRegex(ValueError, 'reference sphere'):
            fit_spherical_bz(self.points*.9, fields(np.full(len(self.points), .04)), 50., self.weights)
        collapsed = np.tile([0., 0., 50.], (100, 1))
        with self.assertRaisesRegex(ValueError, 'do not resolve'):
            fit_spherical_bz(collapsed, fields(np.full(100, .04)), 50., np.ones(100), 4)


class ToothFourierTests(unittest.TestCase):
    def test_actual_28_column_frequency_and_phase_are_resolved(self):
        phi = .13 + np.arange(113)*2*np.pi/113
        bz = .04*(1+7e-6*np.cos(28*phi)-3e-6*np.sin(56*phi))
        report = azimuthal_fourier(phi, bz, expected_orders=(28, 56))
        self.assertGreaterEqual(report['maximum_resolved_order'], 56)
        modes = {row['order']: row for row in report['array_orders']}
        self.assertAlmostEqual(modes[28]['amplitude_ppm'], 7., places=8)
        self.assertAlmostEqual(modes[28]['cosine_ppm'], 7., places=8)
        self.assertAlmostEqual(modes[56]['sine_ppm'], -3., places=8)

    def test_fourier_amplitude_is_distinct_from_peak_to_peak(self):
        phi = np.arange(128)*2*np.pi/128
        bz = .04*(1+10e-6*np.cos(4*phi))
        report = azimuthal_fourier(phi, bz)
        amplitude = report['modes'][3]['amplitude_ppm']
        self.assertAlmostEqual(amplitude, 10., places=8)
        self.assertAlmostEqual(np.ptp(bz)/.04*1e6, 2*amplitude, places=8)


class AnalyticEngine:
    polarization_model = 'synthetic_reference'
    def field(self, design, points_mm):
        p = np.asarray(points_mm)
        return fields(.04*(1+20e-6*(3*p[:, 2]**2-np.sum(p*p, axis=1))/(2*50**2)))


class DomainDispatchTests(unittest.TestCase):
    def setUp(self):
        self.design = Design((RingSpec('R01-L1', 0, 0, 150., 100., 28, 'out', 0.),))
        self.config = default_config()
        self.config['target'].update(geometry='sphere', diameter_mm=100.)

    def test_analyzer_outputs_explicit_sphere_and_resolved_teeth_conventions(self):
        result = analyze_harmonics(self.design, AnalyticEngine(), self.config)
        self.assertTrue(result['spherical']['available'])
        self.assertAlmostEqual(result['spherical']['zonal_ppm']['2'], 20., places=7)
        self.assertEqual([row['degree'] for row in result['spherical']['zonal']], [2, 4, 6, 8])
        self.assertEqual([row['order'] for row in result['azimuthal']['terms']], [28, 56])
        self.assertGreaterEqual(result['spherical']['quadrature']['n_phi'], 113)
        self.assertGreaterEqual(result['azimuthal']['sample_count'], 113)

    def test_disk_produces_no_fake_3d_spectrum_and_writes_readable_plot(self):
        self.config['target']['geometry'] = 'disk'
        with tempfile.TemporaryDirectory() as folder:
            result = analyze_harmonics(self.design, AnalyticEngine(), self.config, output_dir=folder)
            self.assertFalse(result['spherical']['available'])
            self.assertIn('平面', result['spherical']['reason'])
            self.assertTrue(result['azimuthal']['available'])
            self.assertTrue((Path(folder)/'harmonics.json').is_file())
            self.assertGreater((Path(folder)/'harmonics.png').stat().st_size, 1000)

    def test_ball_with_magnetic_source_is_not_called_source_free(self):
        self.config['target']['diameter_mm'] = 400.
        with self.assertRaisesRegex(ValueError, 'no magnetic sources'):
            analyze_harmonics(self.design, AnalyticEngine(), self.config)

    def test_36_column_second_tooth_is_not_aliased_into_low_order_sphere_terms(self):
        class HighOrderEngine:
            polarization_model = 'synthetic_reference'
            def field(self, design, points_mm):
                p = np.asarray(points_mm)
                value = np.real(((p[:, 0]+1j*p[:, 1])/50.)**72)
                return fields(.04*(1+100e-6*value))
        design = Design((RingSpec('R01-L1', 0, 0, 150., 100., 36, 'out', 0.),))
        result = analyze_harmonics(design, HighOrderEngine(), self.config)
        self.assertGreaterEqual(result['spherical']['quadrature']['n_phi'], 145)
        nonconstant = [row for row in result['spherical']['coefficients'] if row['degree']]
        self.assertLess(max(abs(row['coefficient_ppm']) for row in nonconstant), 1e-7)
        self.assertGreater(result['spherical']['residual_rms_ppm'], 1.)
        self.assertAlmostEqual(result['azimuthal']['terms'][1]['amplitude_ppm'], 100., places=7)


class AxisFeatureTests(unittest.TestCase):
    def test_conditioned_fit_recovers_all_even_coefficients_through_z8(self):
        t = np.cos(np.linspace(np.pi, 0., 21))
        coefficients = np.array([.04, 8e-5, -6e-6, 4e-7, -2e-8])
        values = sum(c*t**degree for c, degree in zip(coefficients, (0, 2, 4, 6, 8)))
        fitted, condition = _axis_features(values, t)
        np.testing.assert_allclose(fitted, coefficients, atol=5e-16, rtol=1e-8)
        self.assertLess(condition, 10.)

    def test_synthesis_weights_enable_z6_z8_without_forcing_disk_axial_flatness(self):
        sphere = {'target': {'geometry': 'sphere'}}
        self.assertEqual(_axis_order_weights(sphere, [0, 2, 4, 6, 8]), {2: 1., 4: 1., 6: .25, 8: .1})
        disk = {'target': {'geometry': 'disk'}}
        self.assertEqual(_axis_order_weights(disk, [0, 2, 4, 6, 8]), {})
        sphere['de_novo'] = {'harmonic_max_order': 6, 'harmonic_order_weights': {'6': .5}}
        self.assertEqual(_axis_order_weights(sphere, [0, 2, 4, 6, 8]), {2: 1., 4: 1., 6: .5})
        sphere['de_novo']['harmonic_order_weights']['6'] = -1.
        with self.assertRaises(ValueError):
            _axis_order_weights(sphere, [0, 2, 4, 6, 8])

    def test_basis_cache_and_shape_include_resolved_eighth_order_axis_features(self):
        config = default_config()
        sample = sphere_samples(20., 4, 6)
        with tempfile.TemporaryDirectory() as folder:
            basis = build_basis(config, sample, 5, 24, cache_dir=folder)
            again = build_basis(config, sample, 5, 24, cache_dir=folder)
        self.assertEqual(basis['axis_coefficient_orders'], [0, 2, 4, 6, 8])
        self.assertEqual(basis['axis_A_T_per_T'].shape[0], 21)
        self.assertEqual(basis['axis_coefficients_T_per_T'].shape[0], 5)
        self.assertLess(basis['axis_fit_condition_number'], 10.)
        self.assertTrue(again['cache_hit'])
        np.testing.assert_array_equal(basis['axis_coefficients_T_per_T'], again['axis_coefficients_T_per_T'])


if __name__ == '__main__':
    unittest.main()
