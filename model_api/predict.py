import torch
from fastapi import FastAPI,HTTPException, File, Form, UploadFile
from torchvision import transforms
from pydantic import BaseModel,model_validator
from PIL import Image
import io
from model import word2idx,idx2word,clip_model,CLIPImageCaptioner
import base64
device="cuda" if torch.cuda.is_available() else "cpu"
clip_transform= transforms.Compose([
transforms.Resize((224, 224)),
transforms.ToTensor(),
transforms.Normalize(
    mean=[0.48145466, 0.4578275, 0.40821073],
    std=[0.26862954, 0.26130258, 0.27577711]
)
])
app=FastAPI()
def generate_captions(model:CLIPImageCaptioner,tensor,max_len=50):
    captions=""
    predicted="<S>"
    with torch.no_grad():
        encoder_out=model.encoder(tensor)
        h,c=model.decoder.init_hidden_state(encoder_out)
        while(predicted!="<E>"):
            word_idx_tensor = torch.tensor([word2idx[predicted]], device=device)
            word_embed = model.decoder.embedding(word_idx_tensor).view(1, -1)
            att_out = model.decoder.attention(encoder_out, h)
            if isinstance(att_out, tuple):
                t1, t2 = att_out[0], att_out[1]
                context = t1 if t1.shape[-1] > t2.shape[-1] else t2
            else:
                context = att_out
                
            context = context.view(1, -1)
            lstm_input=torch.cat([word_embed,context],dim=1)
            h,c=model.decoder.lstm_cell(lstm_input,(h,c))
            probs=model.decoder.fc(h)
            predicted_idx=probs.argmax(dim=1).item()
            predicted=idx2word[predicted_idx]
            if predicted not in ["<S>", "<pad>", "<unk>"]:
                captions+=" "+predicted
            if(len(captions.split(" "))>=max_len):
                break
    return captions if not captions.endswith("<E>") else captions[:-3]


def preprocess(photo:bytes)->torch.Tensor:
    raw_images=Image.open(io.BytesIO(photo)).convert("RGB")
    tensor_img=clip_transform(raw_images).unsqueeze(0).to(device)
    return tensor_img
class client(BaseModel):
    userid:int
    photo:str
@app.post("/predict")
def predictimg(user:client):
    #.............................
    #authenticate first
    #..................................
    try:
        photo_bytes=base64.b64decode(user.photo)
        img_tensor=preprocess(photo_bytes)
    except Exception as e:
        raise HTTPException(HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}"))
    captions=generate_captions(clip_model,img_tensor)
    return {
        "userid":user.userid,
        "captions":captions,
        "msg":"request sucessful"
        }

@app.post("/predictwithUploadFile")
async def predictimg(userid: int = Form(...), photo: UploadFile = File(...)):
    # Read bytes directly from request stream
    photo_bytes = await photo.read()
    img_tensor = preprocess(photo_bytes)
    
    captions = generate_captions(clip_model, img_tensor)
    
    return {
        "userid": userid,
        "captions": captions,
        "msg": "request successful"
    }
