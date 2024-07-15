from dashboard.new_app import SensorDashboardApp


# Instantiate and run the app
if __name__ == "__main__":
    app_instance = SensorDashboardApp()
    app_instance.setup_layout()
    app_instance.setup_callbacks()
    app_instance.app.run_server(debug=True)


# from dashboard.data import CustomDashboardData

# t = CustomDashboardData().get_training_windows()

# if __name__ == "__main__":
#     print(f"Sensor Name {t[3][0]}")
#     print(
#         f"Input Features | Shape {len(t[0]), len(t[0][0]), len(t[0][0][0])} | Sample {t[0][0][0]}"
#     )
#     print(
#         f"Labels | Shape {len(t[1]), len(t[1][0]), len(t[1][0][0])} | Sample {t[1][0][0]}"
#     )
#     print(
#         f"Engineered Features | Shape {len(t[2]), len(t[2][0]), len(t[2][0][0]), len(t[2][0][0][0])} | Sample {t[2][0][0][0]}"
#     )
