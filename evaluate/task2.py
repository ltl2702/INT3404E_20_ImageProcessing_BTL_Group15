import pandas as pd
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate the accuracy of a model')
    parser.add_argument('--gt_csv', type=str, required=True, help='Path to the ground truth CSV file')
    parser.add_argument('--pred_csv', type=str, required=True, help='Path to the prediction CSV file')
    return parser.parse_args()



def read_csv(csv_file):
    df = pd.read_csv(csv_file)
    return df

def evaluate_accuracy(gt_csv, pred_csv):
    gt_df = read_csv(gt_csv)
    pred_df = read_csv(pred_csv)
    
    merged_df = pd.merge(gt_df, pred_df, on='image_name', suffixes=('_gt', '_pred'))
    
    # Calculate accuracy
    correct_predictions = (merged_df['label_gt'] == merged_df['label_pred']).sum()
    total_predictions = len(merged_df)
    accuracy = correct_predictions / total_predictions
    
    return accuracy


if __name__ == '__main__':
    args = parse_args()
    accuracy = evaluate_accuracy(args.gt_csv, args.pred_csv)
    print(f'Accuracy: {accuracy}')