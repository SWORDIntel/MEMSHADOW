import pytest
from app.services.hydra.adversarial_suite import adversarial_simulator

@pytest.mark.asyncio
@pytest.mark.security
async def test_run_auth_bypass_simulation():
    """
    Tests the AdversarialSimulator by running the 'auth_bypass' scenario.

    This test validates that the simulator can execute a scenario, run the
    defined attack techniques, and generate a correct report based on the
    (mocked) outcomes.
    """
    # Define the target environment for the simulation
    target_env = "staging"

    # Run the simulation
    report = await adversarial_simulator.simulate_attack(
        scenario_name="auth_bypass",
        target_env=target_env
    )

    # --- Assertions to validate the report ---

    # The scenario name should be correct.
    assert report.scenario_name == "auth_bypass"

    # In our mocked scenario, only one technique (JWTManipulation) is defined.
    assert report.techniques_attempted == 1

    # The JWTManipulationTechnique is hardcoded to return a successful result.
    assert len(report.successful_techniques) == 1

    # The simulation was not blocked by any defense.
    assert report.blocked_at is None

    # Validate the details of the successful technique.
    successful_technique = report.successful_techniques[0]
    assert successful_technique.technique_name == "JWT_manipulation"
    assert successful_technique.successful is True
    assert successful_technique.details["vulnerability"] == "Algorithm confusion (alg=none)"