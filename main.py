from fastapi import FastAPI, Request

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "MCP Test Server Running!"}

@app.post("/message")
async def receive_message(request: Request):
    data = await request.json()
    # Here you can process or store the message
    return {"status": "received", "data": data}