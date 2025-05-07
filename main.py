import torch
from models_Image2Coord import *          # Sketch_Classification is defined here
# from models_Coord2Image import *        # keep commented – not needed with the unified class
from Dataset import get_dataloader
import argparse, os
from datetime import datetime
from tqdm import tqdm
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
now = datetime.now()
dt_string = now.strftime("%d_%m_%Y %H:%M:%S")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sketch Self‑Supervision / Classification')
    parser.add_argument('--dataset_name', type=str, default='QuickDraw',
                        help='TUBerlin or QuickDraw')
    parser.add_argument('--train_mode',
                        choices=['supervised', 'image2coord', 'coord2image'],
                        default='supervised',
                        help="Training type: 'supervised', 'image2coord', or 'coord2image'")
    parser.add_argument('--backbone_name', default='Resnet')
    parser.add_argument('--pool_method', default='AdaptiveAvgPool2d')
    parser.add_argument('--batchsize', type=int, default=64)
    parser.add_argument('--nThreads', type=int, default=8)
    parser.add_argument('--splitTrain', type=float, default=0.9)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--max_epoch', type=int, default=50)
    parser.add_argument('--eval_freq_iter', type=int, default=48000)
    parser.add_argument('--print_freq_iter', type=int, default=50)
    parser.add_argument('--draw_frequency', type=int, default=100)
    parser.add_argument('--drop_rate', type=float, default=0.5)
    parser.add_argument('--encoder_hidden_dim', type=int, default=128)
    parser.add_argument('--decoder_hidden_dim', type=int, default=512)
    parser.add_argument('--print_per_epoch_only', action='store_true',
                    help="Only print metrics once at the end of each epoch instead of every N steps.")

    hp = parser.parse_args()
    hp.fullysupervised = (hp.train_mode == 'supervised')

    dataloader_Train, dataloader_Test = get_dataloader(hp)

    hp.date_time_folder = os.path.join('results', hp.dataset_name, dt_string)
    for sub in ['', 'sketch_Viz', 'models']:
        os.makedirs(os.path.join(hp.date_time_folder, sub), exist_ok=True)

    model = Sketch_Classification(hp).to(device)
    print(f"🚀 Using device: {device}")
    best_acc, step = 0, 0

    def loop_header(epoch, total_epochs, i_batch, step, loss, best_acc):
        print(f"[Epoch {epoch:3d}/{total_epochs}] "
              f"Iter {i_batch:5d} | Step {step:6d} | "
              f"Loss: {loss:8.4f} | Best Acc: {best_acc:6.2f}%")

    if hp.train_mode == 'supervised':
        train_step = model.train_supervised
        eval_fn = model.evaluate
        ckpt_tag = 'modelSupervised'
    elif hp.train_mode == 'image2coord':
        train_step = model.train_Image2Coordinate
        eval_fn = model.fine_tune_linear_LMDB
        ckpt_tag = 'Image2Coordinate'
    else:  # coord2image
        train_step = model.train_Coordinate2Image
        eval_fn = model.fine_tune_linear
        ckpt_tag = 'Coordinate2Image'

    print(f"\n▶️  Starting training for {hp.max_epoch} epochs in '{hp.train_mode}' mode.\n")
    start_time = time.time()

    for epoch in range(hp.max_epoch):
        progress_bar = tqdm(enumerate(dataloader_Train),
                            total=len(dataloader_Train),
                            desc=f"Epoch {epoch+1:3d}/{hp.max_epoch}",
                            leave=False)

        for i_batch, batch in progress_bar:
            loss = train_step(batch, step)
            step += 1

            if not hp.print_per_epoch_only and step % hp.print_freq_iter == 0:
                loop_header(epoch + 1, hp.max_epoch, i_batch, step, loss, best_acc)

            if step % hp.draw_frequency == 0 and 'coord' in hp.train_mode:
                with torch.no_grad():
                    model.evaluate_coordinate_redraw(batch, step)

            if step % hp.eval_freq_iter == 0:
                with torch.no_grad():
                    acc = (eval_fn(dataloader_Train, dataloader_Test)
                           if 'coord' in hp.train_mode else eval_fn(dataloader_Test))
                if acc > best_acc:
                    best_acc = acc
                    torch.save(model.state_dict(),
                               f"{hp.date_time_folder}/models/{ckpt_tag}.pth")

            if step % 20000 == 0:
                torch.save(model.state_dict(),
                           f"{hp.date_time_folder}/models/{ckpt_tag}_{step}.pth")
        if hp.print_per_epoch_only:
            loop_header(epoch + 1, hp.max_epoch, i_batch, step, loss, best_acc)

    total_time = time.time() - start_time
    print(f"\n✅ Training complete in {total_time / 60:.2f} minutes. Best Accuracy: {best_acc:.2f}%\n")
