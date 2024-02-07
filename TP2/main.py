import os
from KL_transform import kl_transform

def question2():
    print('Question 2')

    try:
        os.makedirs('results/question2/')
    except FileExistsError:
        pass

    for image in os.listdir('data'):
        if not image.endswith('.png'):
            continue

        print(f"Processing {image}...")
        kl_transform(os.path.join('data', image), show_img = True, color_space = 'RGB')
        



















if __name__ == '__main__':
    question2()