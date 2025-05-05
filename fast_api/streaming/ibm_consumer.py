from kafka import KafkaConsumer
import json
import ssl
import time
from kafka.errors import NoBrokersAvailable
from fastapi import WebSocket
import asyncio
from global_state import BOOTSTRAP_SERVERS, API_KEY, TOPIC, get_streaming

def ibm_consumer(retries=5, delay=5):
    for attempt in range(retries):
        try:
            consumer = KafkaConsumer(
                TOPIC,  # Topic name
                bootstrap_servers=BOOTSTRAP_SERVERS,
                security_protocol="SASL_SSL",
                sasl_mechanism="PLAIN",
                sasl_plain_username="token",
                sasl_plain_password=API_KEY,  # Replace with your actual API key
                ssl_context=ssl._create_unverified_context(),
                group_id="trading_consumer_group",  # Consumer group ID
                auto_offset_reset="earliest",  # Start reading from the beginning
                enable_auto_commit=True,
                value_deserializer=lambda v: json.loads(v.decode('utf-8'))  # Deserialize JSON messages
            )
            return consumer
        except NoBrokersAvailable as e:
            print(f"Error: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)
    raise Exception("Failed to connect to Kafka after multiple attempts.")

async def ibm_consumer_msg(ws: WebSocket, consumer: any):
    try:
        for message in consumer:
            print(f'consumer : {consumer} : {get_streaming()}')
            if not get_streaming():  # Stop sending if flag is set
                break
            msg = message.value
            print(f'ibm event : {msg}')
            await ws.send_json(msg)
            await asyncio.sleep(2)
    except Exception as e:
        print(f"Error sending data: {e}")
    finally:
        consumer.close()
