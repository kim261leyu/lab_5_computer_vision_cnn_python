import torch
import torch.nn as nn
import time

def train(model, train_loader, val_loader, epochs=10):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    total_train_s = 0.0
    for epoch in range(epochs):
        model.train()
        t_train = time.perf_counter()
        total_loss = 0.0
        for x, y in train_loader:
            optimizer.zero_grad()            
            scores = model(x)   
            loss = loss_fn(scores.view(-1, 10), y.view(-1))
            loss.backward()                  
            optimizer.step()                
            total_loss += loss.item()
        total_train_s += time.perf_counter() - t_train

        model.eval()
        digit_correct = 0
        full_correct = 0
        t_val = time.perf_counter()
        with torch.no_grad():
            for x, y in val_loader:
                scores = model(x)
                predicted = scores.argmax(dim=2)                        # [64, 5]
                digit_correct += (predicted == y).sum().item()          # per-digit count
                full_correct += (predicted == y).all(dim=1).sum().item()  # whole-number count
        
        infer_ms = (time.perf_counter() - t_val) * 1000 / len(val_loader.dataset)                        
        avg_loss = total_loss / len(train_loader)
        
        n = len(val_loader.dataset)
        print(f"epoch {epoch+1}: loss {avg_loss:.4f}, digit acc {digit_correct/(n*5):.3f}, full acc {full_correct/n:.3f}")
    train_ms = (time.perf_counter() - total_train_s) * 1000
    return {
        "digit_acc": digit_correct / (n * 5),
        "full_acc": full_correct / n,
        "train_ms_per_epoch": total_train_s * 1000 / epochs,
        "infer_ms": infer_ms,
    }
    
def ctc_decode(seq):
    result = []
    prev = None
    for v in seq:
        if v != prev and v != 10:
            result.append(v)
        prev = v
    return result
            
def train_crnn(model, train_loader, val_loader, epochs=10):
    loss_fn = nn.CTCLoss(blank=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    total_train_s = 0.0
    for epoch in range(epochs):
        model.train()
        t_train = time.perf_counter()
        total_loss = 0.0
        for x, y in train_loader:
            optimizer.zero_grad()            
            scores = model(x)   
            log_probs = scores.log_softmax(2).permute(1, 0, 2)                  # [40, batch, 11]
            input_lengths = torch.full((x.size(0),), 40, dtype=torch.long)
            target_lengths = torch.full((x.size(0),), 5, dtype=torch.long)
            loss = loss_fn(log_probs, y, input_lengths, target_lengths)
            loss.backward()                  
            optimizer.step()                
            total_loss += loss.item()
        total_train_s += time.perf_counter() - t_train

        model.eval()
        digit_correct = 0
        full_correct = 0
        t_val = time.perf_counter()
        with torch.no_grad():
            for x, y in val_loader:
                scores = model(x)
                pred = scores.argmax(dim=2)                      # [batch, 40]
                for seq, target in zip(pred, y):
                    decoded = ctc_decode(seq.tolist())
                    target = target.tolist()
                    full_correct += (decoded == target)
                    digit_correct += sum(a == b for a, b in zip(decoded, target))
        
        infer_ms = (time.perf_counter() - t_val) * 1000 / len(val_loader.dataset)                        
        avg_loss = total_loss / len(train_loader)
        
        n = len(val_loader.dataset)
        print(f"epoch {epoch+1}: loss {avg_loss:.4f}, digit acc {digit_correct/(n*5):.3f}, full acc {full_correct/n:.3f}")
    
    return {
        "digit_acc": digit_correct / (n * 5),
        "full_acc": full_correct / n,
        "train_ms_per_epoch": total_train_s * 1000 / epochs,
        "infer_ms": infer_ms,
    }