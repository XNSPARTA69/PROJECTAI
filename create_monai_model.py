<<<<<<< HEAD
import torch
from monai.networks.nets import UNet

# Crear el modelo UNet
monai_model = UNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2)
)

# Guardar únicamente los pesos del modelo en un archivo
torch.save(monai_model.state_dict(), "monai_model.pth")
=======
import torch
from monai.networks.nets import UNet

# Crear el modelo UNet
monai_model = UNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2)
)

# Guardar únicamente los pesos del modelo en un archivo
torch.save(monai_model.state_dict(), "monai_model.pth")
>>>>>>> b660f68215c6ef53f11ed0f6667f001a992c4733
print("Archivo 'monai_model.pth' creado exitosamente.")