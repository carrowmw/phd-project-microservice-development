# phd_project/dashboard/__main__.py

from .app import SensorDashboardApp


# Instantiate and run the app
if __name__ == "__main__":
    app_instance = SensorDashboardApp()
    app_instance.setup_layout()
    app_instance.setup_callbacks()
    app_instance.app.run_server(debug=True)
