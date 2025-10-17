from playwright.sync_api import sync_playwright, expect
import traceback

def run_verification():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        try:
            # Navigate to the Streamlit app
            page.goto("http://localhost:8501", timeout=90000)

            # Wait for the page to settle
            page.wait_for_timeout(5000)

            # Verify the skill level selector is present
            expect(page.locator("[data-testid='stSelectbox']").first).to_be_visible()

            # Select "🌱 Complete Beginner"
            page.locator("[data-testid='stSelectbox']").first.click()
            page.locator("li[id^='st-']").get_by_text("🌱 Complete Beginner").click()

            # Take a screenshot of the initial state
            page.screenshot(path="jules-scratch/verification/phase1_initial.png")

            # Go to the OLS tab and compute the solution
            page.get_by_text("📊 Ordinary Least Squares").click()
            page.get_by_role("button", name="⚡ Compute OLS Solution").click()
            page.wait_for_load_state("networkidle", timeout=60000)
            page.wait_for_timeout(5000)

            # Verify that advanced metrics are hidden for beginners
            expect(page.get_by_text("Adjusted R²")).not_to_be_visible()
            page.screenshot(path="jules-scratch/verification/phase1_beginner_metrics.png")

            # Verify the simplified coefficient visualization
            expect(page.get_by_text("How much does each feature matter?")).to_be_visible()
            page.screenshot(path="jules-scratch/verification/phase1_beginner_viz.png")

            print("Verification script completed successfully.")

        except Exception as e:
            print(f"An error occurred during verification: {e}")
            print(traceback.format_exc())

        finally:
            browser.close()

if __name__ == "__main__":
    run_verification()