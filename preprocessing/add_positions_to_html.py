from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from pathlib import Path
import shutil

from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.common.by import By
import wandb


NUM_WORKERS = 32
TIMEOUT = 15


def process_file(input_dir: Path, output_dir: Path, filename: str):
    output_file = output_dir / filename
    if output_file.exists():
        return

    options = webdriver.FirefoxOptions()
    options.headless = True
    options.set_capability('pageLoadStrategy', 'eager')
    options.set_preference('javascript.enabled', False)

    with webdriver.Firefox(options=options) as driver:
        driver.set_page_load_timeout(TIMEOUT)

        url = 'file://' + str((input_dir / filename).absolute())

        try:
            driver.get(url)
        except:
            pass

        for element in driver.find_elements(By.TAG_NAME, '*'):
            position_information = {**element.location, **element.size}

            for key, value in position_information.items():
                # Add data-x, data-y, etc. to the DOM tree
                driver.execute_script('arguments[0].setAttribute(arguments[1],arguments[2])', element, f'data-{key}', value)

        html = driver.execute_script('return document.documentElement.outerHTML')

    output_file.parent.mkdir(exist_ok=True, parents=True)

    with open(output_file, 'w') as _file:
        _file.write(html)


def main():
    wandb.init(
        project='information_extraction',
        name='random-split',
        job_type='preprocess',
    )

    dataset_artifact = wandb.use_artifact('swde:latest')

    input_dir = Path(dataset_artifact.download(root=os.path.expanduser('~/Data/SWDE/')))
    output_dir = Path('~/Data/SWDE-pos/').expanduser()

    target_filenames = []
    num_existing = 0

    for parent_dir, _, filenames in os.walk(input_dir):
        parent_path = Path(parent_dir)
        for filename in filenames:
            if filename.endswith('.htm'):
                file_path = (parent_path / filename).relative_to(input_dir)

                if (output_dir / file_path).exists():
                    num_existing += 1
                else:
                    target_filenames.append(file_path)

    with tqdm(desc=f'File', total=len(target_filenames) + num_existing, initial=num_existing, smoothing=0) as pbar:
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures_to_filenames = {}
            for filename in target_filenames:
                future = executor.submit(
                    process_file,
                    input_dir,
                    output_dir,
                    filename,
                )
                future.add_done_callback(lambda _: pbar.update())
                futures_to_filenames[future] = filename

            for future in as_completed(futures_to_filenames):
                filename = futures_to_filenames[future]
                try:
                    future.result()
                except Exception as e:
                    tqdm.write(f'Exception for {filename}: {e}')

    shutil.copytree(input_dir / 'groundtruth', output_dir / 'groundtruth', dirs_exist_ok=True)

    artifact = wandb.Artifact('swde-pos', type='dataset',
                              description='An augmented version of the SWDE dataset, containing position '
                                          'information in each HTML tag')
    artifact.add_dir(str(output_dir))

    wandb.log_artifact(artifact)
    wandb.finish()


if __name__ == '__main__':
    main()
