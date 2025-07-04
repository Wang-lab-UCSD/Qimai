import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patheffects as path_effects

def get_learning_curve(file_AUCs):
    
    df = pd.read_csv(file_AUCs, sep='\t')

    train_loss = df['Loss_train'].tolist()
    val_loss = df['Loss_val'].tolist()
    val_acc = df['accuracy'].tolist()
    val_auc = df['ROC'].tolist()
    val_prc = df['PRC'].tolist()
    val_f1 = df['F1'].tolist()
    val_mcc = df['MCC'].tolist()
    
    # visualize the loss as the network trained
    fig,(ax1,ax2) = plt.subplots(2,1, sharex=True)
    fig.set_size_inches(5,5)

    ax1.plot(train_loss, label='Training')
    ax1.set_ylabel('loss')
    ax1.plot(val_loss,color='tab:orange', label='Validation')
    minposs = val_loss.index(min(val_loss))

    ax2.plot(val_auc,color='tab:blue', label='ROC')
    ax2.plot(val_acc,color='tab:orange', label='accuracy')
    ax2.plot(val_prc,color='tab:green', label='PRC')
    ax2.plot(val_f1,color='tab:red', label='F1')
    ax2.plot(val_mcc,color='tab:gray', label='MCC')
    ax2.set_ylabel('metric')
    ax2.set_xlabel('epoch')

    ax1.grid()
    ax2.grid()
    fig.legend(bbox_to_anchor = (1,0.5), loc = 'center left')
    fig.tight_layout()
    fig.savefig(file_AUCs.replace('txt','pdf'), bbox_inches='tight')


def get_metric_df(filename):
    df = pd.read_csv(filename, sep='\t')
    df = df[(df['PRC']>0) & (df['MCC']>0)]
    
    # Check which grouping column exists
    if 'protein' in df.columns:
        group_col = 'protein'
    elif 'dna' in df.columns:
        group_col = 'dna'
    else:
        raise ValueError("Neither 'protein' nor 'dna' column found in the data")
    
    df2 = df.groupby(group_col).mean()
    df2.reset_index(inplace=True)
    df2 = df2.rename(columns={'index': group_col})

    df3 = pd.melt(df2, id_vars=[group_col], value_vars=['accuracy', 'sensitivity', 'specificity', 'ROC', 'PRC', 'MCC', 'F1'])
    print(df3.shape)
    print(df3.head())
    return df3

def add_median_labels(ax: plt.Axes, fmt: str = ".2f") -> None:
    """Add text labels to the median lines of a seaborn boxplot.

    Args:
        ax: plt.Axes, e.g. the return value of sns.boxplot()
        fmt: format string for the median value
    """
    lines = ax.get_lines()
    boxes = [c for c in ax.get_children() if "Patch" in str(c)]
    start = 4
    if not boxes:  # seaborn v0.13 => fill=False => no patches => +1 line
        boxes = [c for c in ax.get_lines() if len(c.get_xdata()) == 5]
        start += 1
    lines_per_box = len(lines) // len(boxes)
    for median in lines[start::lines_per_box]:
        x, y = (data.mean() for data in median.get_data())
        # choose value depending on horizontal or vertical plot orientation
        value = x if len(set(median.get_xdata())) == 1 else y
        text = ax.text(x, y, f'{value:{fmt}}', ha='center', va='center',
                       fontweight='bold', color='white')
        # create median-colored border around white text for contrast
        text.set_path_effects([
            path_effects.Stroke(linewidth=3, foreground=median.get_color()),
            path_effects.Normal(),
        ])

def get_metric_plot(df,filename):
    sns.set(font_scale=1.5)
    sns.set_style(style='ticks')
    plt.rcParams['figure.figsize'] = [6,5]
    fig, ax = plt.subplots()
    # fig.suptitle("inference on 160 TFs")

#     ax = sns.violinplot(data=df, x = "variable", y = "value", inner=None)
#     PROPS = {
#         'boxprops':{'facecolor':'white', 'edgecolor':'black', 'zorder':2},
#     }

    ax = sns.boxplot(data=df, x="variable",y="value", width=0.8)
    ax.set(ylim=(0,1))
    ax.tick_params(axis='x',rotation=45)
    ax.set(xlabel=None)
    ax.set(ylabel=None)
    add_median_labels(ax)
#     # add median values to the plot
#     medians = df.groupby("variable")["value"].median()
#     for x_loc, y_loc in enumerate(medians):
#         ax.text(x_loc, y_loc, f'{y_loc:.2f}', color='red', ha="center", va="center", fontweight='bold')

#     fig.legend(bbox_to_anchor = (1,0.5), loc = 'center left')
    fig.tight_layout()
    # plt.yticks([0.4, 0.6, 0.8, 0.9 ,1])
    fig.savefig(filename.replace('.txt','_plot.pdf'), bbox_inches='tight')
