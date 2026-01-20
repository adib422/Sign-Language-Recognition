import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tqdm import tqdm

import onnx
import onnxruntime as ort


MODEL_NAME = "mobilenet_v2_grayscale"
NUM_CLASSES = 36 
EPOCHS = 10
LEARNING_RATE = 0.001
PATIENCE = 5


OUTPUT_DIR = "output_models"
BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model.pth")
FINAL_MODEL_PATH = os.path.join(OUTPUT_DIR, "final_model.pth")
ONNX_MODEL_PATH = os.path.join(OUTPUT_DIR, "asl_model.onnx")

os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device('cpu')
print(f"Using device: {DEVICE}")


class ASLMobileNet(nn.Module):
    
    def __init__(self, num_classes=36):
        super(ASLMobileNet, self).__init__()

        self.mobilenet = models.mobilenet_v2(pretrained=True)
        
        original_conv = self.mobilenet.features[0][0]
        self.mobilenet.features[0][0] = nn.Conv2d(
            1,  # Input: 1 channel (grayscale)
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False
        )
        
        in_features = self.mobilenet.classifier[1].in_features
        self.mobilenet.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.mobilenet(x)


def build_model(num_classes):

    model = ASLMobileNet(num_classes=num_classes)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model: MobileNetV2 (Grayscale)")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Input: [B, 1, 224, 224] (grayscale)")
    print(f"Output: [B, {num_classes}]")
    
    return model



# TRAINING
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 
                         'acc': f'{100.*correct/total:.2f}%'})
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def train_model(model, train_loader, val_loader, epochs, lr, patience, device):
    print(f"\n{'='*60}")
    print("TRAINING MODEL")
    print(f"{'='*60}")
    print(f"Epochs: {epochs} | LR: {lr} | Patience: {patience}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2
    )
    
    best_val_acc = 0.0
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    start_time = time.time()
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 60)
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        scheduler.step(val_acc)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"✓ Best model saved! (Val Acc: {val_acc:.2f}%)")
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.2f} minutes")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    model.load_state_dict(torch.load(BEST_MODEL_PATH))
    return model, history



# EVALUATION

def evaluate_model(model, test_loader, device, idx_to_class):
    print(f"\n{'='*60}")
    print("EVALUATION")
    print(f"{'='*60}")
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"\n✓ Test Accuracy: {accuracy*100:.2f}%")
    
    class_names = [idx_to_class[i] for i in sorted(idx_to_class.keys())]
    print(f"\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    cm = confusion_matrix(all_labels, all_preds)
    return accuracy, cm, class_names, all_preds, all_labels


# ONNX EXPORT
def export_to_onnx(model, onnx_path, input_shape=(1, 1, 224, 224)):
    print(f"\n{'='*60}")
    print("EXPORTING TO ONNX")
    print(f"{'='*60}")
    
    model.eval()

    dummy_input = torch.randn(input_shape)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"✓ ONNX model saved to: {onnx_path}")
    
    # Verify ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model verified successfully")
    
    # Get model size
    model_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"✓ ONNX model size: {model_size_mb:.2f} MB")
    
    return onnx_path


def test_onnx_inference(onnx_path, test_loader, idx_to_class):

    print(f"\n{'='*60}")
    print("TESTING ONNX INFERENCE")
    print(f"{'='*60}")
    
    # Load ONNX model
    ort_session = ort.InferenceSession(onnx_path)
    
    # Test on a few batches
    correct = 0
    total = 0
    inference_times = []
    
    for i, (images, labels) in enumerate(test_loader):
        if i >= 10:  # Test on 10 batches
            break
        
        # Prepare input
        ort_inputs = {ort_session.get_inputs()[0].name: images.numpy()}
        
        # Inference
        start_time = time.time()
        ort_outputs = ort_session.run(None, ort_inputs)
        inference_time = (time.time() - start_time) * 1000  # ms
        inference_times.append(inference_time)
        
        # Get predictions
        predictions = np.argmax(ort_outputs[0], axis=1)
        
        correct += (predictions == labels.numpy()).sum()
        total += labels.size(0)
    
    accuracy = 100. * correct / total
    avg_inference_time = np.mean(inference_times)
    
    print(f"✓ ONNX Accuracy (10 batches): {accuracy:.2f}%")
    print(f"✓ Avg inference time: {avg_inference_time:.2f} ms/batch")
    print(f"✓ Avg inference per image: {avg_inference_time/32:.2f} ms")



# VISUALIZATION

def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(history['train_acc'], label='Train', linewidth=2)
    ax1.plot(history['val_acc'], label='Validation', linewidth=2)
    ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(history['train_loss'], label='Train', linewidth=2)
    ax2.plot(history['val_loss'], label='Validation', linewidth=2)
    ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_history.png'), dpi=150)
    print(f"Saved training_history.png")
    plt.close()


def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=150)
    print(f"Saved confusion_matrix.png")
    plt.close()


# MAIN PIPELINE

def main():
    print("\n" + "="*60)
    print("ASL SIGN LANGUAGE DETECTION")
    print("="*60)

    if not os.path.exists('dataset_metadata.pt'):
        raise FileNotFoundError("Run Phase 1&2 first to create dataset_metadata.pt")
    
    metadata = torch.load('dataset_metadata.pt')
    idx_to_class = metadata['idx_to_class']
    num_classes = metadata['num_classes']
    
    print(f"Loaded metadata: {num_classes} classes")
    
    import sys
    sys.path.insert(0, '.')
    from dataset_prep import prepare_data
    
    train_loader, val_loader, test_loader, _ = prepare_data()
    
    model = build_model(num_classes)
    model = model.to(DEVICE)
    

    # Train model
    model, history = train_model(
        model, train_loader, val_loader,
        EPOCHS, LEARNING_RATE, PATIENCE, DEVICE
    )
    
    torch.save(model.state_dict(), FINAL_MODEL_PATH)
    print(f"\n✓ PyTorch model saved: {FINAL_MODEL_PATH}")
    
    # Evaluate
    accuracy, cm, class_names, all_preds, all_labels = evaluate_model(
        model, test_loader, DEVICE, idx_to_class
    )
    
    # Plot results
    print(f"\nGenerating visualizations...")
    plot_training_history(history)
    plot_confusion_matrix(cm, class_names)
    
    # Export to ONNX
    onnx_path = export_to_onnx(model, ONNX_MODEL_PATH, input_shape=(1, 1, 224, 224))
    
    # Test ONNX inference
    test_onnx_inference(onnx_path, test_loader, idx_to_class)
    
    # Save results
    results = {
        'test_accuracy': accuracy,
        'num_classes': num_classes,
        'model_name': MODEL_NAME,
        'idx_to_class': idx_to_class,
        'history': history,
        'onnx_path': onnx_path
    }
    torch.save(results, os.path.join(OUTPUT_DIR, 'results.pt'))
    
    print(f"\n{'='*60}")
    print("✓ TRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"\nFinal Results:")
    print(f"  Test Accuracy: {accuracy*100:.2f}%")
    print(f"  PyTorch model: {FINAL_MODEL_PATH}")
    print(f"  ONNX model: {ONNX_MODEL_PATH}")
    print(f"\nFiles in '{OUTPUT_DIR}/':")
    print(f"  - asl_model.onnx (ONNX format - use this for inference!)")
    print(f"  - best_model.pth (PyTorch checkpoint)")
    print(f"  - final_model.pth (PyTorch final)")
    print(f"  - results.pt (metrics)")
    print(f"  - training_history.png")
    print(f"  - confusion_matrix.png")


if __name__ == "__main__":
    main()