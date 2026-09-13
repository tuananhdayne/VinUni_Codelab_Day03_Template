"""
Lab #3: Baseline Chatbot vs ReAct Agent.

The implementation is intentionally offline so it works without an API key.
"""

import json
from typing import Any, Dict, List, Tuple

from tools import TOOL_DEFINITIONS, TOOL_MAP


SYSTEM_PROMPT = """Bạn là một ReAct Agent thông minh hỗ trợ khách hàng Vingroup.
Bạn chỉ sử dụng các công cụ sau:
{tools}

Quy trình trả lời bắt buộc:
Thought: <Suy nghĩ bước tiếp theo>
Action: {{"name": "<tên tool>", "args": {{<tham số>}}}}
Observation: <Kết quả từ tool>
... (Lặp lại cho tới khi có đủ dữ liệu)
Final Answer: <Câu trả lời hoàn chỉnh cho khách hàng>
"""


class ChatbotBaseline:
    """Baseline chatbot that does not use tools."""

    def query(self, user_input: str) -> Dict[str, Any]:
        return {
            "answer": f"[Chatbot Baseline] Trả lời cho: {user_input}",
            "tool_calls": [],
            "status": "success",
            "mode": "offline",
        }


class ReActAgent:
    """Offline ReAct agent using the local tool registry."""

    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.trace: List[Dict[str, Any]] = []

    @staticmethod
    def parse_city_code(text: str) -> str:
        text_upper = text.upper()
        for code in ("SGN", "HAN", "DAD"):
            if code in text_upper:
                return code
        if "HÀ NỘI" in text_upper:
            return "HAN"
        if "HỒ CHÍ MINH" in text_upper or "SÀI GÒN" in text_upper:
            return "SGN"
        if "ĐÀ NẴNG" in text_upper or "ĐÀ NANG" in text_upper:
            return "DAD"
        return "SGN"

    @staticmethod
    def _price_limit(user_lower: str) -> int:
        if "2 triệu" in user_lower or "2.000.000" in user_lower:
            return 2_000_000
        if "1.5 triệu" in user_lower or "1,5 triệu" in user_lower:
            return 1_500_000
        if "500k" in user_lower:
            return 500_000
        return 5_000_000

    @staticmethod
    def _flight_answer(origin: str, destination: str, flights: List[Dict[str, Any]], limit: int) -> str:
        if not flights:
            return f"Không tìm thấy chuyến bay nào từ {origin} đi {destination} dưới {limit:,} VND."
        lines = [
            f"- {flight['airline']} ({flight['flight_number']}): "
            f"{flight['departure_time']} - Giá: {flight['price_vnd']:,} VNĐ"
            for flight in flights
        ]
        return f"Tìm thấy {len(flights)} chuyến bay từ {origin} đi {destination}:\n" + "\n".join(lines)

    def plan_and_execute_step(self, user_input: str, iteration: int) -> Tuple[str, bool]:
        user_lower = user_input.lower()

        if "chính sách" in user_lower or "đổi trả" in user_lower:
            thought = "Đây là câu hỏi FAQ chung về chính sách. Không cần sử dụng tool."
            answer = (
                "Vé máy bay Vinpearl có thể hỗ trợ đổi ngày trước 24 giờ so với giờ "
                "khởi hành, phí đổi vé là 350.000 VNĐ/vé cộng chênh lệch giá vé (nếu có)."
            )
            self.trace.append({"iteration": iteration, "thought": thought, "final_answer": answer})
            return answer, True

        needs_flight = any(keyword in user_lower for keyword in ("chuyến bay", "vé", "bay từ", "vé máy bay"))
        needs_weather = any(keyword in user_lower for keyword in ("thời tiết", "mặc gì", "nhiệt độ", "mưa"))

        if needs_flight and iteration == 1:
            origin = "HAN" if "han" in user_lower or "hà nội" in user_lower else "DAD"
            destination = "DAD" if "dad" in user_lower or "đà nẵng" in user_lower else "SGN"
            limit = self._price_limit(user_lower)
            action = {
                "name": "get_flight_info",
                "args": {"origin": origin, "destination": destination, "max_price": limit},
            }
            observation = TOOL_MAP[action["name"]](**action["args"])
            self.trace.append({
                "iteration": iteration,
                "thought": f"Tra cứu chuyến bay từ {origin} đi {destination} với giá tối đa {limit} VND.",
                "action": action,
                "observation": observation,
            })
            if not needs_weather:
                return self._flight_answer(origin, destination, observation, limit), True
            return "Đã tra cứu chuyến bay; tiếp tục kiểm tra thời tiết.", False

        if needs_weather and (iteration == 2 or (iteration == 1 and not needs_flight)):
            city_code = self.parse_city_code(user_input)
            action = {"name": "get_weather_forecast", "args": {"city_code": city_code}}
            observation = TOOL_MAP[action["name"]](**action["args"])
            self.trace.append({
                "iteration": iteration,
                "thought": f"Kiểm tra thông tin thời tiết tại {city_code}.",
                "action": action,
                "observation": observation,
            })
            if not needs_flight:
                answer = (
                    f"Thời tiết tại {observation.get('city', city_code)}: "
                    f"{observation.get('temperature_c', 'N/A')}°C, "
                    f"{observation.get('condition', '')}.\n"
                    f"Gợi ý: {observation.get('recommendation', '')}"
                )
                return answer, True
            return "Đã kiểm tra thời tiết; tổng hợp kết quả.", False

        flight_trace = next(
            (step for step in self.trace if step.get("action", {}).get("name") == "get_flight_info"),
            None,
        )
        weather_trace = next(
            (step for step in self.trace if step.get("action", {}).get("name") == "get_weather_forecast"),
            None,
        )
        flights = flight_trace["observation"] if flight_trace else []
        weather = weather_trace["observation"] if weather_trace else {}
        flight_summary = "Không tìm thấy chuyến bay phù hợp."
        if flights:
            flight_summary = "\n".join(
                f"- {flight['airline']} ({flight['flight_number']}): "
                f"{flight['departure_time']} - Giá: {flight['price_vnd']:,} VNĐ"
                for flight in flights
            )
        answer = (
            f"1. Thông tin chuyến bay:\n{flight_summary}\n\n"
            f"2. Thông tin thời tiết & trang phục:\n"
            f"- Thời tiết tại {weather.get('city', 'địa phương')}: "
            f"{weather.get('temperature_c', 'N/A')}°C ({weather.get('condition', '')}).\n"
            f"- Gợi ý trang phục: {weather.get('recommendation', '')}"
        )
        self.trace.append({
            "iteration": iteration,
            "thought": "Đã thu thập đủ thông tin để trả lời khách hàng.",
            "final_answer": answer,
        })
        return answer, True

    def run(self, user_input: str) -> Dict[str, Any]:
        self.trace = []
        for iteration in range(1, self.max_iterations + 1):
            answer, is_final = self.plan_and_execute_step(user_input, iteration)
            if is_final:
                return {
                    "answer": answer,
                    "trace": self.trace,
                    "iterations": iteration,
                    "status": "completed",
                }
        return {
            "answer": "Lỗi: Agent đã vượt quá số bước lặp tối đa (Max Iterations Safeguard).",
            "trace": self.trace,
            "iterations": self.max_iterations,
            "status": "max_iterations_reached",
        }


def main() -> None:
    user_query = "Tìm cho tôi chuyến bay từ HAN đi SGN dưới 2 triệu, rồi cho biết thời tiết SGN nên mặc gì?"

    print("=== RUNNING CHATBOT BASELINE ===")
    print(ChatbotBaseline().query(user_query))

    print("\n=== RUNNING REACT AGENT ===")
    agent = ReActAgent(max_iterations=5)
    result = agent.run(user_query)
    print("Result:", result)
    print("Trace Log:", json.dumps(agent.trace, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
