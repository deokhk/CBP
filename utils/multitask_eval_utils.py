
def extract_schema_prediction_labels(seq):
    if "<table>" in seq:
        table_labels = seq.split("<table>")[1].split("<column>")[0].split(",")
        table_labels = [x.strip() for x in table_labels if x != ""]
    else:
        table_labels = [""]

    if "<column>" in seq:
        column_labels = seq.split("<column>")[1].split(",")
        column_labels = [x.strip() for x in column_labels if x != ""]
    else:
        column_labels = [""]

    return table_labels, column_labels


def extract_schema_prediction_labels_batch(batch_seq):
    batch_table_labels = []
    batch_column_labels = []
    for seq in batch_seq:
        table_labels, column_labels = extract_schema_prediction_labels(seq)
        batch_table_labels.append(table_labels)
        batch_column_labels.append(column_labels)
    return batch_table_labels, batch_column_labels

def extract_value_prediction_labels(seq):
    if "<value>" in seq:
        value_labels = seq.split("<value>")[1].split(",")
        value_labels = [x.strip() for x in value_labels if x != ""]
    else:
        value_labels = [""]
        
    return value_labels 

def extract_value_prediction_labels_batch(batch_seq):
    batch_value_labels = []
    for seq in batch_seq:
        value_label = extract_value_prediction_labels(seq)
        batch_value_labels.append(value_label)
    return batch_value_labels


def calculate_recall(predictions, references):
    # Count the number of correctly predicted items
    correct_predictions = sum(1 for pred in predictions if pred in references)
    
    # Total number of relevant items
    total_references = len(references)
    
    # Calculate recall
    recall = correct_predictions / total_references if total_references > 0 else 0.0
    
    return recall

def calculate_precision(predictions, references):
    # Count the number of correctly predicted items
    correct_predictions = sum(1 for pred in predictions if pred in references)
    
    # Total number of predicted items
    total_predictions = len(predictions)
    
    # Calculate precision
    precision = correct_predictions / total_predictions if total_predictions > 0 else 0.0
    
    return precision

def calculate_f1_score(predictions, references):
    # Calculate precision
    precision = calculate_precision(predictions, references)
    
    # Calculate recall
    recall = calculate_recall(predictions, references)
    
    # Calculate F1 score
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return f1_score

def batch_compute_f1(batch_predictions, batch_references):
    batch_f1 = []
    for predictions, references in zip(batch_predictions, batch_references):
        # Calculate f1 score
        f1 = calculate_f1_score(predictions, references)
        batch_f1.append(f1)
    avg_batch_f1 = sum(batch_f1) / len(batch_f1)
    return avg_batch_f1
