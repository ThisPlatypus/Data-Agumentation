from Mode import MyDDPM, MyUNet, generate_new_images
import torch 
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'

fold = '/home/chiara/DataAUG/DATA/MAL_dataset/TR_HEX'

for classd in os.listdir(fold):
    print(classd)
# Loading the trained model
    best_model = MyDDPM(MyUNet(), n_steps=1000, device=device)
    best_model.load_state_dict(torch.load(f'/home/chiara/DataAUG/MOD/DIFF/{classd}.pth', map_location=device))
    best_model.eval()
    print("Model loaded")
    print("Generating new images")
    for i in range(0,700):
        generated = generate_new_images(
                best_model,
                LLL=i,
                n_samples=100,
                device=device,
                gif_name=classd
            )
    print('finish')
