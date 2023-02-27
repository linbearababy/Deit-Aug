@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    ## Define augmentation functions (add)
    tta_transforms = [torchvision.transforms.RandomResizedCrop(224),
                      torchvision.transforms.RandomHorizontalFlip(),
                      torchvision.transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                      torchvision.transforms.RandomGrayscale(p=0.2),
                      torchvision.transforms.RandomRotation(15),
                      torchvision.transforms.RandomPerspective(),
                      torchvision.transforms.RandomAffine(15)]

    # switch to evaluation mode
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        ## apply test-time augmentation (add)
        output_list = []
        for tta_transform in tta_transforms:
            tta_image = tta_transform(images)
            with torch.cuda.amp.autocast():
                output = model(tta_image)
            output_list.append(output)

        # take the average of model predictions from different augmentations
        output = torch.mean(torch.stack(output_list), dim=0)

        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
