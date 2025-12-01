import pytest
from app.services.hydra.adversarial_suite import adversarial_simulator, AdversarialSimulator

@pytest.mark.asyncio
@pytest.mark.security
async def test_run_auth_bypass_simulation():
    """
    Tests the AdversarialSimulator by running the 'auth_bypass' scenario.

    This test validates that the simulator can execute a scenario, run the
    defined attack techniques, and generate a correct report.
    """
    # Define the target environment for the simulation (must not be production)
    target_env = "staging.example.com"

    # Run the simulation
    report = await adversarial_simulator.simulate_attack(
        scenario="auth_bypass",
        target_env=target_env
    )

    # --- Assertions to validate the report ---

    # The scenario name should be correct
    assert report.scenario == "auth_bypass"

    # Should have attempted at least one technique
    assert report.techniques_attempted >= 1

    # Duration should be positive
    assert report.duration_seconds >= 0

    # Either succeeded or was blocked
    assert report.successful_techniques or report.blocked_at


@pytest.mark.asyncio
@pytest.mark.security
async def test_production_protection():
    """Test that simulations cannot run against production"""
    simulator = AdversarialSimulator()

    with pytest.raises(ValueError, match="production"):
        await simulator.simulate_attack(
            scenario="auth_bypass",
            target_env="https://production.example.com"
        )

    with pytest.raises(ValueError, match="production"):
        await simulator.simulate_attack(
            scenario="auth_bypass",
            target_env="https://prod.example.com"
        )


@pytest.mark.asyncio
@pytest.mark.security
async def test_invalid_scenario():
    """Test handling of invalid scenarios"""
    simulator = AdversarialSimulator()

    with pytest.raises(ValueError, match="Unknown scenario"):
        await simulator.simulate_attack(
            scenario="invalid_scenario",
            target_env="staging.example.com"
        )


@pytest.mark.asyncio
@pytest.mark.security
async def test_jwt_manipulation_technique():
    """Test JWT manipulation attack technique execution"""
    from app.services.hydra.adversarial_suite import JWTManipulation

    technique = JWTManipulation()
    result = await technique.execute("staging.example.com")

    assert result.technique_name == "JWT_Manipulation"
    assert result.blocked is True
    assert "timestamp" in result.model_dump()


@pytest.mark.asyncio
@pytest.mark.security
async def test_session_hijacking_technique():
    """Test session hijacking attack technique execution"""
    from app.services.hydra.adversarial_suite import SessionHijacking

    technique = SessionHijacking()
    result = await technique.execute("staging.example.com")

    assert result.technique_name == "Session_Hijacking"
    assert result.blocked is True
