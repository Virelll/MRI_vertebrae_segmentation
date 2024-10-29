from sklearn.model_selection import train_test_split

def train_test_split_data(images,masks,shape):
    img_train, img_test, masks_train, masks_test = train_test_split(images, masks, test_size=0.2, random_state=42,
                                                                                    shuffle=True)
    img_train = img_train.reshape((-1, 1, shape, shape))
    masks_train = masks_train.reshape((-1, 1, shape, shape))
    img_test = img_test.reshape((-1, 1, shape, shape))
    masks_test = masks_test.reshape((-1, 1, shape, shape))
    return img_train, img_test, masks_train, masks_test
