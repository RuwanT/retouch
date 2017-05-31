from custom_networks import retouch_dual_net


def train_model(model):
    # create something to generate parches
    print "nothing"

if __name__ == "__main__":
    model = retouch_dual_net(input_shape=(224,224,3))
    model.compile(optimizer='sgd')
    train_model(model)