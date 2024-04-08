import matplotlib.pyplot as plt

def visualise_img_pair(img1, img2, figsize=(3,6)):
    #
    # Visualise a pair of images side by side
    #
    plt.figure(figsize=figsize)
    plt.subplot(1,2,1)
    plt.imshow(img1,aspect="auto", cmap='gray', vmin=0, vmax=1)
    plt.title('Original', fontsize=8)
    plt.xticks([])
    plt.yticks([])
    plt.subplot(1,2,2)
    plt.imshow(img2,aspect="auto", cmap='gray', vmin=0, vmax=1)
    plt.title('Reconstructed', fontsize=8)
    plt.xticks([])
    plt.yticks([])
    plt.show()