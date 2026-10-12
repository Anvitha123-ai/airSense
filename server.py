import http.server
import socketserver
import json
import csv
import os

PORT = 8000
DATA_CSV = os.path.join(os.path.dirname(__file__), "data", "raw", "city_day.csv")

class AirSenseHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/api/data":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            
            data = {}
            if os.path.exists(DATA_CSV):
                with open(DATA_CSV, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        city = row.get("City", "Unknown")
                        if city not in data:
                            data[city] = []
                        # Cast numbers
                        parsed_row = {
                            "Date": row.get("Date"),
                            "City": city,
                            "PM2.5": float(row.get("PM2.5", 0) or 0),
                            "PM10": float(row.get("PM10", 0) or 0),
                            "NO2": float(row.get("NO2", 0) or 0),
                            "SO2": float(row.get("SO2", 0) or 0),
                            "CO": float(row.get("CO", 0) or 0),
                            "O3": float(row.get("O3", 0) or 0),
                            "Temperature": float(row.get("Temperature", 0) or 0),
                            "Humidity": float(row.get("Humidity", 0) or 0),
                            "AQI": int(float(row.get("AQI", 0) or 0)),
                            "AQI_Bucket": row.get("AQI_Bucket", "Moderate")
                        }
                        data[city].append(parsed_row)
            
            self.wfile.write(json.dumps(data).encode("utf-8"))
            return
            
        return super().do_GET()

    def do_POST(self):
        if self.path == "/api/risk-assessment":
            content_length = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_length)
            try:
                import risk_assessment
                payload = json.loads(post_data.decode("utf-8"))
                age = float(payload.get("age", 30))
                aqi = float(payload.get("aqi", 150))
                pm25 = float(payload.get("pm25", 60))
                pm10 = float(payload.get("pm10", 100))
                hours = float(payload.get("exposure_hours", 2.0))
                activity = str(payload.get("activity_level", "Moderate"))
                conds = list(payload.get("selected_conditions", []))

                result = risk_assessment.calculate_personal_risk(
                    age=age, aqi=aqi, pm25=pm25, pm10=pm10,
                    exposure_hours=hours, activity_level=activity,
                    selected_conditions=conds
                )
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps(result).encode("utf-8"))
                return
            except Exception as e:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
                return

        return super().do_POST()

if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), AirSenseHandler) as httpd:
        print(f"AirSense web server running at: http://localhost:{PORT}")
        httpd.serve_forever()
