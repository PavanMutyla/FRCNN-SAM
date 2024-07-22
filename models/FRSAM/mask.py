import matplotlib.pyplot as plt

def show_mask(masks,image, random_color = True):
    '''
    Takes predicted masks , image and return the overlayed image.
    random_color: If true, can have multiple colors for same calss instance
                : else, one color for all classes and instances, which is  ""Dodger Blue""
    '''

    plt.figure(figsize=(10, 10))
    plt.imshow(image)

    for mask in masks:
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        plt.savefig('res.png')
        plt.imshow(mask_image)
