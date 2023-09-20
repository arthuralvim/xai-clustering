import matplotlib.pyplot as plt


def plot_relevance(
    image_1,
    image_2,
    image_1_title="Ground Truth",
    image_2_title="Layer Wise Relevance Propagation",
    cmap="seismic",
):
    def add_colobar(image, image_ax):
        box = image_ax.get_position()
        plt.colorbar(
            image,
            cax=plt.axes([box.x0 + box.width * 1.02, box.y0, 0.01, box.height]),
            orientation="vertical",
            label="Pixel Relevance",
        )

    f, ax = plt.subplots(1, 2, figsize=(9, 9))
    plt.subplots_adjust(wspace=0, hspace=0)

    ax[0].imshow(image_1)
    ax[0].title.set_text(image_1_title)

    im1 = ax[1].imshow(image_2, cmap=cmap)
    ax[1].title.set_text(image_2_title)

    for a in ax:
        a.axis("off")

    add_colobar(im1, ax[1])
    plt.show()


def analyze_example(image, label, image_lwrp, pred_label):
    print("Ground Truth for this image: ", label)
    print("Prediction was: ", pred_label)
    plot_relevance(
        image,
        image_lwrp,
        image_1_title="Ground Truth",
        image_2_title="Layer Wise Relevance Propagation",
        cmap="seismic",
    )
    if not (label == pred_label):
        print("This image is not classified correctly.")
