"""High-level 6-DoF parafoil model.

This module provides a typed orchestration layer around the pure functions in
``dynamics.py``. It is intentionally API-focused (simulation state management,
flap commands, and time stepping) while keeping force/moment math in pure
functions for testability.
"""

from __future__ import annotations

import numpy as np

from .dynamics import DynamicsDiagnostics, compute_derivatives, update_flaps
from .frames import body_to_inertial
from .types import AeroCoefficients, Inputs, PhysicalParams, State, StateDerivative


class ParafoilModel6DOF:
    """Typed stateful simulator used by guidance and identification code.

    Parameters
    ----------
    params
        Either a :class:`PhysicalParams` instance or a partial mapping of
        parameter overrides.
    initial_state
        Initial state as :class:`State` or sequence
        ``[position, velocity_body, eulers, angular_velocity]``.
    initial_inputs
        Initial :class:`Inputs`. If omitted, zero flap and zero wind are used.
    coefficients
        Optional aerodynamic coefficient overrides (typed object, mapping,
        or ordered list).
    """

    def __init__(
        self,
        params: dict | PhysicalParams,
        initial_state: State | list[np.ndarray],
        initial_inputs: Inputs | None = None,
        coefficients: AeroCoefficients | dict[str, float] | list[float] | None = None,
    ) -> None:
        self.params = params if isinstance(params, PhysicalParams) else PhysicalParams.from_mapping(params)
        self.coeffs = coefficients if isinstance(coefficients, AeroCoefficients) else AeroCoefficients()
        if not isinstance(coefficients, AeroCoefficients):
            self.coeffs.update(coefficients)

        self.state = initial_state if isinstance(initial_state, State) else State.from_sequence(initial_state)
        self.inputs = initial_inputs if isinstance(initial_inputs, Inputs) else Inputs(0.0, 0.0, np.zeros(3))

        # Actuator model state (actual flap position can lag target commands).
        self.flap_left = float(self.inputs.flap_left)
        self.flap_right = float(self.inputs.flap_right)
        self.flap_left_target = float(self.inputs.flap_left)
        self.flap_right_target = float(self.inputs.flap_right)
        self.delta_a = 0.0
        self.delta_s = 0.0

        self.last_state_derivative: StateDerivative | None = None
        self.last_diagnostics: DynamicsDiagnostics | None = None
        self.error_flag = False
        self.evaluate()

    def set_coefficients(self, values: dict[str, float] | list[float] | None) -> None:
        """Update aerodynamic coefficient values.

        Parameters
        ----------
        values
            Mapping or ordered list of aerodynamic coefficient values.
            ``None`` leaves coefficients unchanged.
        """
        self.coeffs.update(values)
        self.evaluate()

    def set_state(self, state: State | list[np.ndarray]) -> None:
        """Set the current simulation state.

        Parameters
        ----------
        state
            State object or sequence
            ``[position, velocity_body, eulers, angular_velocity]``.
        """
        self.state = state if isinstance(state, State) else State.from_sequence(state)
        self.evaluate()

    def set_wind(self, wind_inertial: np.ndarray) -> None:
        """Set ambient wind in inertial coordinates.

        Parameters
        ----------
        wind_inertial
            Wind vector in m/s with shape ``(3,)``.
        """
        self.inputs = Inputs(self.inputs.flap_left, self.inputs.flap_right, wind_inertial)
        self.evaluate()

    def set_flap_targets(self, flap_left: float, flap_right: float) -> None:
        """Set desired flap target angles.

        Parameters
        ----------
        flap_left
            Left flap command in radians.
        flap_right
            Right flap command in radians.
        """
        self.flap_left_target = float(flap_left)
        self.flap_right_target = float(flap_right)

    def set_inputs(self, inputs: Inputs) -> None:
        """Set flap targets and wind from a typed input object.

        Parameters
        ----------
        inputs
            Input container with flap commands and inertial wind.
        """
        self.flap_left_target = float(inputs.flap_left)
        self.flap_right_target = float(inputs.flap_right)
        self.inputs = Inputs(inputs.flap_left, inputs.flap_right, inputs.wind_inertial)

    def advance_flaps(self, dt: float) -> None:
        """Advance actuator states toward flap targets.

        Parameters
        ----------
        dt
            Time increment in seconds.
        """
        self.flap_left, self.flap_right, self.delta_a, self.delta_s = update_flaps(
            flap_l=self.flap_left,
            flap_r=self.flap_right,
            flap_l_desired=self.flap_left_target,
            flap_r_desired=self.flap_right_target,
            dt=float(dt),
            flap_time_constant=self.params.flap_time_constant,
        )

    def evaluate(self, state: State | None = None) -> tuple[StateDerivative, DynamicsDiagnostics]:
        """Compute derivatives and diagnostics.

        Parameters
        ----------
        state
            Optional override state for evaluation. If omitted, the model's
            internal state is evaluated and cached.

        Returns
        -------
        tuple[StateDerivative, DynamicsDiagnostics]
            State derivative and diagnostics at the evaluation point.
        """
        eval_state = self.state if state is None else state
        eval_inputs = Inputs(self.flap_left, self.flap_right, self.inputs.wind_inertial)
        state_dot, diag = compute_derivatives(
            state=eval_state,
            inputs=eval_inputs,
            params=self.params,
            coeffs=self.coeffs,
            delta_a=self.delta_a,
            delta_s=self.delta_s,
        )
        if state is None:
            self.last_state_derivative = state_dot
            self.last_diagnostics = diag
            self.error_flag = diag.singularity_warning or diag.invalid_airspeed_warning
        return state_dot, diag

    def derivative_sequence(self, state_sequence: list[np.ndarray]) -> list[np.ndarray]:
        """Solver adapter for sequence-based ODE routines.

        Parameters
        ----------
        state_sequence
            Sequence form of state ordered as
            ``[position, velocity_body, eulers, angular_velocity]``.

        Returns
        -------
        list[np.ndarray]
            Sequence-form derivative aligned with the input ordering.
        """
        state = State.from_sequence(state_sequence)
        state_dot, _ = self.evaluate(state=state)
        return state_dot.as_sequence()

    def step(self, dt: float) -> State:
        """Advance the model one integration step using RK4.

        Parameters
        ----------
        dt
            Time increment in seconds.

        Returns
        -------
        State
            Updated state after integration.
        """
        self.advance_flaps(dt)
        x0 = self.state.as_sequence()

        k1 = self.derivative_sequence(x0)
        k2 = self.derivative_sequence([x + 0.5 * dt * dx for x, dx in zip(x0, k1)])
        k3 = self.derivative_sequence([x + 0.5 * dt * dx for x, dx in zip(x0, k2)])
        k4 = self.derivative_sequence([x + dt * dx for x, dx in zip(x0, k3)])

        x_next = [
            x + (dt / 6.0) * (d1 + 2.0 * d2 + 2.0 * d3 + d4)
            for x, d1, d2, d3, d4 in zip(x0, k1, k2, k3, k4)
        ]
        self.state = State.from_sequence(x_next)
        self.evaluate()
        return self.state

    @property
    def inertial_position(self) -> np.ndarray:
        """Current position in world coordinates.

        Returns
        -------
        np.ndarray
            World-frame position vector with ``z`` positive upward.
        """
        ned_to_world = np.array([1.0, 1.0, -1.0])
        return self.params.initial_pos + ned_to_world * self.state.position

    @property
    def inertial_velocity(self) -> np.ndarray:
        """Current inertial-frame velocity.

        Returns
        -------
        np.ndarray
            Velocity vector rotated from body to inertial frame.
        """
        if self.last_diagnostics is None:
            self.evaluate()
        return body_to_inertial(self.state.velocity_body, self.last_diagnostics.cdm, inverse=False)

    @property
    def euler_rates(self) -> np.ndarray:
        """Current Euler angle rates.

        Returns
        -------
        np.ndarray
            ``[phi_dot, theta_dot, psi_dot]`` in rad/s.
        """
        if self.last_diagnostics is None:
            self.evaluate()
        return self.last_diagnostics.euler_rates.copy()
