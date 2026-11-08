"""Gradient-based unlearning primitives and their request optimizer."""
import math
import random
import time
import torch
import torch.nn.functional as F
from utils import encode

def train_request(model, reference, tokenizer, text, retain, algorithm, args, deadline):
    chunks = [text[i:i+args.chunk_size] for i in range(0, len(text), args.chunk_size)]
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95),
                                weight_decay=0.1, foreach=False)
    total = max(1, args.epochs*len(chunks))
    warmup = int(0.1*total)
    def scale(step):
        if step < warmup:
            return step/max(1, warmup)
        return 0.1 + 0.45*(1+math.cos(math.pi*(step-warmup)/max(1,total-warmup)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, scale)
    updates = 0
    model.train()
    ga_methods={'GA':Gradient_ascent,'GA_GD':Gradient_ascent_descent,'GA_KL':Gradient_ascent_KL}
    if algorithm in ga_methods:
        if time.perf_counter()>=deadline: return 0
        retain_texts=[row['text'] for row in retain]
        if algorithm=='GA':
            ga_methods[algorithm](model,tokenizer,text,optimizer,scheduler,args.epochs,args.chunk_size,model.device)
        elif algorithm=='GA_GD':
            ga_methods[algorithm](model,tokenizer,text,retain_texts,optimizer,scheduler,args.epochs,args.chunk_size,model.device)
        else:
            ga_methods[algorithm](model,reference,tokenizer,text,retain_texts,optimizer,scheduler,args.epochs,args.chunk_size,model.device)
        model.zero_grad(set_to_none=True)
        return args.epochs*len(chunks)
    for _ in range(args.epochs):
        for index, text_chunk in enumerate(chunks):
            if time.perf_counter() >= deadline:
                model.zero_grad(set_to_none=True)
                return updates
            inputs = encode(tokenizer, text_chunk, model.device, args.chunk_size)
            if inputs is None:
                continue
            optimizer.zero_grad(set_to_none=True)
            if algorithm == "RL":
                labels = torch.randint(model.config.vocab_size, inputs.input_ids.shape,
                                       device=model.device)
                loss = model(**inputs, labels=labels).loss
                loss.backward()
            else:
                # Sequence-summed NPO log ratio, matching the existing beta=0.1.
                with torch.no_grad():
                    oracle = reference(**inputs, labels=inputs.input_ids).loss
                current = model(**inputs, labels=inputs.input_ids).loss
                ratio = (current-oracle)*(inputs.input_ids.shape[1]-1)
                loss = -F.logsigmoid(0.1*ratio)*20
                loss.backward()
                if algorithm == "NPO_KL":
                    retained = retain[index % len(retain)]["text"]
                    # Fixed character window per forget chunk; retain membership
                    # uses the same unrequested-record retain set.
                    retained = retained[:args.chunk_size]
                    rinputs = encode(tokenizer, retained, model.device, args.chunk_size)
                    with torch.no_grad():
                        ref_logits = reference(**rinputs).logits[:, :-1].float()
                        target = ref_logits.softmax(-1)
                    logits = model(**rinputs).logits[:, :-1].float()
                    kl = F.kl_div(logits.log_softmax(-1), target,
                                  reduction="sum") / target.shape[1]
                    kl.backward()
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            optimizer.step()
            scheduler.step()
            updates += 1
    model.zero_grad(set_to_none=True)
    return updates


def Gradient_ascent(model, tokenizer, forget_data, optimizer, scheduler, epochs, chunk_size, device):
    """
    Perform gradient ascent optimization for unlearning.

    Args:
        model: The model to be optimized
        tokenizer: Tokenizer for processing input data
        forget_data: Dataset to forget (list of texts)
        optimizer: Optimizer (e.g., AdamW)
        scheduler: Learning rate scheduler
        device: Device, either 'cuda' or 'cpu'
        epochs: Number of training epochs (default: 3)
        chunk_size: Batch size (default: 16)

    Returns:
        None
    """
    data_length = len(forget_data)
    num_chunks = math.ceil(data_length / chunk_size)

    for epoch in range(epochs):
        print(f"\n[Unlearning] Epoch {epoch + 1} ...")
        optimizer.zero_grad()
        total_loss = 0.0

        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = start_idx + chunk_size
            chunk_text = forget_data[start_idx:end_idx]

            # Tokenize inputs
            inputs = tokenizer(
                chunk_text,
                return_tensors="pt",
                truncation=True,
                max_length=chunk_size,
                padding=True
            ).to(device)

            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            labels = input_ids.masked_fill(attention_mask == 0, -100)

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            weighted_loss = -loss  # Apply negative loss for gradient ascent

            # Accumulate total loss and backpropagate
            total_loss += weighted_loss.item()
            weighted_loss.backward()
            optimizer.step()
            scheduler.step()

        avg_loss = total_loss / num_chunks
        print(f"[Epoch {epoch + 1}] weighted_loss={avg_loss:.4f}")


def Gradient_ascent_descent(model, tokenizer, forget_data, retain_set, optimizer, scheduler, epochs, chunk_size, device):
    """
    Perform a mixed gradient ascent (forget set) and gradient descent (retain set)
    optimization process for unlearning.

    Args:
        model: The model to be optimized
        tokenizer: Tokenizer for processing input data
        forget_data: Dataset to forget (list of texts)
        retain_set: Dataset to retain (list of texts)
        optimizer: Optimizer (e.g., AdamW)
        scheduler: Learning rate scheduler
        device: Device, either 'cuda' or 'cpu'
        epochs: Number of training epochs
        chunk_size: Batch size for training
        gamma: (not implemented here) typically a weighting factor for the retain loss, default is 1.0

    Returns:
        None
    """
    data_length_forget = len(forget_data)
    random.shuffle(retain_set)
    # Merge multiple strings from retain_set into one large retain_data
    retain_data = " ".join(retain_set)
    data_length_retain = len(retain_data)

    # Calculate number of batches for forget data
    num_chunks_forget = math.ceil(data_length_forget / chunk_size)

    for epoch in range(epochs):
        print(f"\n[Unlearning] Epoch {epoch + 1} ...")
        model.train()  # Set model to training mode
        optimizer.zero_grad()
        total_loss = 0.0

        # Process forget_data (gradient ascent)
        for chunk_idx in range(num_chunks_forget):
            start_idx = chunk_idx * chunk_size
            end_idx = start_idx + chunk_size
            forget_chunk_text = forget_data[start_idx:end_idx]

            # Tokenize forget batch
            forget_inputs = tokenizer(
                forget_chunk_text,
                return_tensors="pt",
                truncation=True,
                max_length=chunk_size,
                padding=True
            ).to(device)

            forget_input_ids = forget_inputs["input_ids"]
            forget_attention_mask = forget_inputs["attention_mask"]
            forget_labels = forget_input_ids.masked_fill(forget_attention_mask == 0, -100)

            # Forward pass on forget data
            forget_outputs = model(
                input_ids=forget_input_ids,
                attention_mask=forget_attention_mask,
                labels=forget_labels
            )
            loss_forget = forget_outputs.loss
            loss_forget = -loss_forget  # Negate loss for gradient ascent (forgetting)

            # Tokenize retain batch
            retain_chunk_text = retain_data[start_idx:end_idx]
            retain_inputs = tokenizer(
                retain_chunk_text,
                return_tensors="pt",
                truncation=True,
                max_length=chunk_size,
                padding=True
            ).to(device)

            retain_input_ids = retain_inputs["input_ids"]
            retain_attention_mask = retain_inputs["attention_mask"]
            retain_labels = retain_input_ids.masked_fill(retain_attention_mask == 0, -100)

            # Forward pass on retain data
            retain_outputs = model(
                input_ids=retain_input_ids,
                attention_mask=retain_attention_mask,
                labels=retain_labels
            )
            loss_retain = retain_outputs.loss

            # Combined loss: ascent on forget + descent on retain
            loss = loss_forget + loss_retain

            # Track both losses
            total_loss += loss_retain.item()
            total_loss += loss_forget.item()

            # Backpropagation and update
            loss.backward()
            optimizer.step()
            scheduler.step()

        # Compute average loss for the epoch
        avg_loss = total_loss / (num_chunks_forget + 1)
        print(f"[Epoch {epoch + 1}] avg_loss={avg_loss:.4f}")


def kl_loss(prob_p, prob_q):
    return -(prob_p * torch.log(prob_q + 1e-12)).sum(-1).mean()


def Gradient_ascent_KL(model, infer_model, tokenizer, forget_data, retain_set, optimizer, scheduler, epochs, chunk_size, device):
    """
    Perform a mixed gradient ascent (forget set) and gradient descent (retain set)
    optimization process for unlearning.

    Args:
        model: The model to be optimized
        infer_model: Reference model (oracle) for guiding retain optimization
        tokenizer: Tokenizer for processing input data
        forget_data: Dataset to forget (list of texts)
        retain_set: Dataset to retain (list of texts)
        optimizer: Optimizer (e.g., AdamW)
        scheduler: Learning rate scheduler
        device: Device, either 'cuda' or 'cpu'
        epochs: Number of training epochs
        chunk_size: Batch size for training
        gamma: (not implemented here) typically a weighting factor for retain loss, default is 1.0

    Returns:
        None
    """
    data_length_forget = len(forget_data)
    random.shuffle(retain_set)
    # Merge multiple strings from retain_set into one large retain_data
    retain_data = " ".join(retain_set)

    # Calculate number of forget batches
    num_chunks_forget = math.ceil(data_length_forget / chunk_size)

    for epoch in range(epochs):
        print(f"\n[Unlearning] Epoch {epoch + 1} ...")
        model.train()  # Set model to training mode
        optimizer.zero_grad()
        total_loss = 0.0

        # Process forget_data (gradient ascent)
        for chunk_idx in range(num_chunks_forget):
            start_idx = chunk_idx * chunk_size
            end_idx = start_idx + chunk_size
            forget_chunk_text = forget_data[start_idx:end_idx]

            # Tokenize forget batch
            forget_inputs = tokenizer(
                forget_chunk_text,
                return_tensors="pt",
                truncation=True,
                max_length=chunk_size,
                padding=True
            ).to(device)

            forget_input_ids = forget_inputs["input_ids"]
            forget_attention_mask = forget_inputs["attention_mask"]
            forget_labels = forget_input_ids.masked_fill(forget_attention_mask == 0, -100)

            # Forward pass on forget data
            forget_outputs = model(
                input_ids=forget_input_ids,
                attention_mask=forget_attention_mask,
                labels=forget_labels
            )
            forget_loss = forget_outputs.loss
            loss_forget = -forget_loss  # Negate loss for gradient ascent (forgetting)

            # Tokenize retain batch
            retain_chunk_text = retain_data[start_idx:end_idx]
            retain_inputs = tokenizer(
                retain_chunk_text,
                return_tensors="pt",
                truncation=True,
                max_length=chunk_size,
                padding=True
            ).to(device)

            retain_input_ids = retain_inputs["input_ids"]
            retain_attention_mask = retain_inputs["attention_mask"]
            retain_labels = retain_input_ids.masked_fill(retain_attention_mask == 0, -100)

            # Forward pass on retain data
            retain_outputs = model(
                input_ids=retain_input_ids,
                attention_mask=retain_attention_mask,
                labels=retain_labels
            )
            with torch.no_grad():
                infer_retain_outputs = infer_model(
                    input_ids=retain_input_ids,
                    attention_mask=retain_attention_mask,
                    labels=retain_labels
                )

            # KL divergence between model and oracle retain predictions
            prob_retain_p = torch.softmax(retain_outputs.logits, dim=-1)
            prob_retain_q = torch.softmax(infer_retain_outputs.logits, dim=-1)
            loss_retain = kl_loss(prob_retain_p, prob_retain_q)

            # Combined loss: ascent on forget + descent on retain
            loss = loss_forget + loss_retain

            # Track both losses
            total_loss += loss_retain.item()
            total_loss += loss_forget.item()

            # Backpropagation and update
            loss.backward()
            optimizer.step()
            scheduler.step()

        # Compute average loss for the epoch
        avg_loss = total_loss / (num_chunks_forget + 1)
        print(f"[Epoch {epoch + 1}] avg_loss={avg_loss:.4f}")
