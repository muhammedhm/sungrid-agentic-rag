from app.tools import check_eligibility_tool


def _invoke(**kwargs):
    return check_eligibility_tool.invoke(kwargs)


def test_eligible_household():
    result = _invoke(
        household_zip="94101",
        annual_income_usd=80_000,
        system_size_kw=5.0,
        installer_approved=True,
    )
    assert result["eligible"] is True
    assert result["estimated_rebate_usd"] == 2000.0  # 5000W * 0.40


def test_rebate_capped_at_4000():
    result = _invoke(
        household_zip="94101",
        annual_income_usd=80_000,
        system_size_kw=10.0,
        installer_approved=True,
    )
    assert result["eligible"] is True
    assert result["estimated_rebate_usd"] == 4000.0  # 10kW * 0.40 = 4000, at the cap


def test_ineligible_installer_not_approved():
    result = _invoke(
        household_zip="94101",
        annual_income_usd=80_000,
        system_size_kw=5.0,
        installer_approved=False,
    )
    assert result["eligible"] is False
    assert "installer" in result["reason"].lower()


def test_ineligible_zip_outside_service_region():
    result = _invoke(
        household_zip="10001",
        annual_income_usd=80_000,
        system_size_kw=5.0,
        installer_approved=True,
    )
    assert result["eligible"] is False
    assert "zip" in result["reason"].lower()


def test_ineligible_income_over_threshold():
    result = _invoke(
        household_zip="94101",
        annual_income_usd=150_000,
        system_size_kw=5.0,
        installer_approved=True,
    )
    assert result["eligible"] is False
    assert "income" in result["reason"].lower()


def test_ineligible_system_size_out_of_range():
    result = _invoke(
        household_zip="94101",
        annual_income_usd=80_000,
        system_size_kw=2.0,
        installer_approved=True,
    )
    assert result["eligible"] is False
    assert "size" in result["reason"].lower()


def test_hard_check_order_installer_fails_even_with_bad_zip_too():
    # installer check should short-circuit first regardless of other failures
    result = _invoke(
        household_zip="10001",
        annual_income_usd=150_000,
        system_size_kw=2.0,
        installer_approved=False,
    )
    assert result["eligible"] is False
    assert "installer" in result["reason"].lower()
