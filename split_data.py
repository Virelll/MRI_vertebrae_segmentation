from sklearn.model_selection import train_test_split

def train_test_split_data(images,masks,shape):
    img_train, img_test, masks_train, masks_test = train_test_split(images, masks, test_size=0.2, random_state=42,
                                                                                    shuffle=True)
    img_train = img_train.reshape((-1, 1, shape, shape))
    masks_train = masks_train.reshape((-1, 1, shape, shape))
    img_test = img_test.reshape((-1, 1, shape, shape))
    masks_test = masks_test.reshape((-1, 1, shape, shape))
    return img_train, img_test, masks_train, masks_test



# img_train_384, img_test_384, masks_train_384, masks_test_384 = train_test_split(images_384, masks_384,
#                                                                                 test_size = 0.2, random_state = 42,
#                                                                                 shuffle=True)
#
# img_train_384 = img_train_384.reshape((-1, 1, 384, 384))
# masks_train_384 = masks_train_384.reshape((-1, 1, 384, 384))
# img_test_384 = img_test_384.reshape((-1, 1, 384, 384))
# masks_test_384 = masks_test_384.reshape((-1, 1, 384, 384))
#
# img_train_320, img_test_320, masks_train_320, masks_test_320 = train_test_split(images_320, masks_320,
#                                                                                 test_size = 0.2, random_state = 42,
#                                                                                 shuffle=True)
#
# img_train_320 = img_train_320.reshape((-1, 1, 320, 320))
# masks_train_320 = masks_train_320.reshape((-1, 1, 320, 320))
# img_test_320 = img_test_320.reshape((-1, 1, 320, 320))
# masks_test_320 = masks_test_320.reshape((-1, 1, 320, 320))