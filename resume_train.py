from main_train import *

if __name__ == '__main__':
    print("Building the dataset...")
    solo_head, resnet50_fpn, train_loader, test_loader = build_dataset()
    
    # In the last argument of train() put the name of the 
    # checkpoint file you want to resume training from
    print("Training...")
    losses, focal_losses, dice_losses = train(solo_head, resnet50_fpn, train_loader, test_loader, 'solo_epoch_0')
    
    # Plot the loss curves
    print("Plotting loss curves...")
    plot_loss_curves(losses, focal_losses, dice_losses)