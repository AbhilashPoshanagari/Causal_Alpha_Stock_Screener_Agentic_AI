from fastapi import WebSocket
import asyncio
from kafka import KafkaConsumer
from global_state import LOCAL_KAFKA_BROKER, TOPIC, get_streaming

def get_consumer():
    consumer = KafkaConsumer(
            TOPIC,
            bootstrap_servers=LOCAL_KAFKA_BROKER,
            auto_offset_reset="earliest",
            enable_auto_commit=True
        )
    return consumer
async def consumer_msg(ws: WebSocket, consumer: any):
    try:
        for message in consumer:
            if not get_streaming():  # Stop sending if flag is set
                break
            msg = message.value.decode("utf-8")
            print(f'kafka : {msg}')
            await ws.send_text(msg)
            await asyncio.sleep(2)
    except Exception as e:
        print(f"Error sending data: {e}")
    finally:
        consumer.close()

# async def send_static_data(websocket: WebSocket, is_running: bool):
#     """Send static data to WebSocket clients every second."""
#     try:
#         while is_running:
#             data = {"price": 500, "timestamp": 1709112345}  # Static data
#             print(f"📤 Sent: {data}")  # Debugging output
#             await websocket.send_json(data)  # Send JSON data
#             await asyncio.sleep(1)  # Send data every second
#     except Exception as e:
#         print(f"❌ Error sending data: {e}")
#     finally:
#         print("⚠️ Stopping data stream")