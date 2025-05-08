import asyncio
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from streaming.kafka_consumer import consumer_msg, get_consumer
from kafka import KafkaConsumer
from global_state import set_streaming, get_streaming, get_mlflow_server, set_mlflow_server, path
from fastapi.middleware.cors import CORSMiddleware
from streaming.ibm_consumer import ibm_consumer, ibm_consumer_msg
from lstm_predict import lstm_prediction, get_all_registered_versions, getparameters_metrics
from extract_features import get_features
import mlflow
import pandas as pd
from fastapi.responses import JSONResponse
from mlflow.tracking import MlflowClient
from agent_ai import predict_llm
import asyncio
from fastapi.staticfiles import StaticFiles
import os
from fastapi import HTTPException
from datetime import datetime
from team_2_stock_recommendation_agentic_v6_all_agent_logged import gradio_predict

app = FastAPI()
app.mount("/content", StaticFiles(directory=os.path.abspath("content")), name="content")
CONTENT_DIR = Path(os.path.abspath("content"))


def get_file_metadata(file_path: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} does not exist")

    stats = os.stat(file_path)
    relative_path = file_path.relative_to(CONTENT_DIR)
    url_path = f"/content/{relative_path.as_posix()}"

    return {
        "file_path": url_path,
        "size_bytes": stats.st_size,
        "last_modified": datetime.fromtimestamp(stats.st_mtime).isoformat()
    }

client = MlflowClient()
origins:list = [
    "http://127.0.0.1:4200",
    "http://localhost:4200",
    "http://localhost:8080",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# try:
#     consumer = get_consumer()
#     server = 'local'
#     print(f'local consumer : {consumer}')
# except:
#     try:
#         consumer = ibm_consumer()
#         server = 'ibm'
#         print(f'ibm consumer : {consumer}')
#     except:
#         consumer = False
#         server = 'NoBrokersAvailable'
#         print(f'consumer : {consumer}')
    
# Store connected users
connected_clients = set()

class User(BaseModel):
    username: str
    password: str

class IP(BaseModel):
    mlflow_ip: str

class ModelParams(BaseModel):
    tag: str
    version: str
    aliases: str
    source: str
    run_id: str

# Define request schema
class PredictRequest(BaseModel):
    stock_ticker: str
    analysis_type: str = "final_synthesis"
    n_runs: int = 1
    check_similarity: bool = False
    similarity_threshold: float = 0.9
    compare_fields: bool = False
    use_cache: bool = True
    show_similarity_summary: bool = False
    add_weights: bool = False
    macro_weight_val: float = 0.0
    sector_weight_val: float = 0.0
    tech_weight_val: float = 0.0

@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI!"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "query": q}

@app.post("/users/")
def get_all_users(user: User):
    return {"user_count": 20, "request": user}

@app.post("/mlflow")
def get_all_users(ip: IP):
    ip = set_mlflow_server(ip.mlflow_ip)
    mlflow.set_tracking_uri(ip)
    return JSONResponse(content={
        "status": 200,
        "data": f"ML flow connected to {ip}"
    })

@app.post("/llm/predict")
async def llm_predict(req: PredictRequest):
    try:
        # print(f"data : ", req)
        response = await asyncio.to_thread(
            gradio_predict,
            req.stock_ticker, req.analysis_type, req.n_runs, req.check_similarity, req.similarity_threshold,
            req.compare_fields, req.use_cache, req.show_similarity_summary, req.add_weights,
            req.macro_weight_val, req.sector_weight_val, req.tech_weight_val)
        print("eval path : ", response[4], " sim path : ", response[5])
        try:
            eval_log_file = get_file_metadata(response[4])
            sim_log_file = get_file_metadata(response[5])
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))

        return JSONResponse(content={"status": 200, "data": {"macro_analysis": response[0],
                                 "sector_and_company_analysis": response[1],
                                 "technical_analysis": response[2],
                                 "final_synthesis": response[3],
                                 "eval_log_path": eval_log_file,
                                 "sim_log_path": sim_log_file,
                                 "price_chart": response[6],
                                 "bollinger_chart": response[7],
                                 "rsi_chart": response[8],
                                 "volatility_chart": response[9],
                                 "macd_chart": response[10]}})
    except Exception as e:
        return JSONResponse(content={"status": 500, "error": str(e)})

@app.get("/mlflow/getallversions/{model}")
def get_versions(model: str):
    versions_data = get_all_registered_versions(model_name=model)
    return JSONResponse(content={
        "status": 200,
        "data": versions_data
    })

@app.get("/mlflow/predict")
def predict_stock_prices():
    features, date_values, actual_prices = get_features()
    predicted_prices = lstm_prediction(feature=features, mlflow=mlflow)
      # Convert date values to timestamps (milliseconds)
    formatted_data = []
    for date, actual, predicted in zip(date_values, actual_prices, predicted_prices):
        formatted_data.append({"Date":date, "Actual_price":actual, "Predicted_price":predicted[0]})

    return JSONResponse(content={
        "status": 200,
        "data": formatted_data
    })

@app.get("/mlflow/predict/{model}/{run_id}")
def predict_stock_prices(model:str, run_id: str):
    features, date_values, actual_prices = get_features()
    # model_uri = "models:/" + model_name + '@' + tag
    # print(f"Model uri : {model_uri}")
    predicted_prices = lstm_prediction(feature=features, mlflow=mlflow, model=model, run_id=run_id)
      # Convert date values to timestamps (milliseconds)
    formatted_data = []
    params, metrics = getparameters_metrics(run_id)
    for date, actual, predicted in zip(date_values, actual_prices, predicted_prices):
        formatted_data.append({"Date":date, "Actual_price":actual, "Predicted_price":predicted[0]})

    return JSONResponse(content={
        "status": 200,
        "data": {"values": formatted_data, "params": params, "metrics": metrics}
    })

@app.get("/datasets/trained_data")
def get_trained_data():
    data = pd.read_csv(path + 'dataset/base_models/nifty50.csv')
    trained_data = data[:-100].to_dict(orient="records")
    return JSONResponse(content={"data":trained_data, "status":200})




# @app.get('/kafka/start')
# async def start_kafka_stream():
#     """API to start WebSocket streaming."""
#     if not get_streaming():
#         is_running = True
#         set_streaming(is_running)
#         if server == 'local':
#             for ws in connected_clients:
#                 print("Starting WebSocket streaming task...")
#                 asyncio.create_task(consumer_msg(ws=ws, consumer=consumer))
#         elif server == 'ibm':
#             for ws in connected_clients:
#                 print("Starting WebSocket streaming task...")
#                 asyncio.create_task(ibm_consumer_msg(ws=ws, consumer=consumer))
#         else:
#             return {"message": "NoBrokersAvailable"}
        
#     return {"message": "WebSocket streaming started"}

# @app.get("/kafka/stop")
# async def stop_stream():
#     """API to stop WebSocket streaming."""
#     is_running = False  # Stop sending data
#     set_streaming(is_running)
#     # Notify clients that the stream has stopped
#     for ws in connected_clients:
#         await ws.send_json({"message": "WebSocket streaming stopped"})

#     return {"message": "WebSocket streaming stopped"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)
    print("New WebSocket client connected!")
    try:
        while True:
            data = await websocket.receive_text()  # Keep connection alive
    except WebSocketDisconnect:
        print("WebSocket client disconnected!")
        connected_clients.remove(websocket)
    finally:
        await websocket.close()

