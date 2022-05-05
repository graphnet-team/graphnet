import os
import pandas as pd
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from graphnet.models.task.reconstruction import BinaryClassificationTask
import torch
from torch.optim.adam import Adam
from graphnet.components.loss_functions import  BinaryCrossEntropyLoss
from graphnet.components.loss_functions import  LogCoshLoss, VonMisesFisher2DLoss, EuclideanDistance
from graphnet.components.utils import fit_scaler
from graphnet.data.constants import FEATURES, TRUTH
from graphnet.data.utils import get_equal_proportion_neutrino_indices
from graphnet.models import Model
from graphnet.models.detector.icecube import IceCubeUpgrade
from graphnet.models.gnn import DynEdge_V3
from graphnet.models.graph_builders import KNNGraphBuilder
from graphnet.models.task.reconstruction import PassOutput1, PassOutput3, ZenithReconstructionWithKappa
from graphnet.models.training.callbacks import ProgressBar, PiecewiseLinearLR
from graphnet.models.training.utils import get_predictions, make_train_validation_dataloader, save_results
import dill

def save_results(db, tag, results, archive,model, validation_loss = None, training_loss = None):
    db_name = db.split('/')[-1].split('.')[0]
    path = archive + '/' + db_name + '/' + tag
    os.makedirs(path, exist_ok = True)
    results.to_csv(path + '/results.csv')
    model.save(path + '/' + tag + '.pth')
    #torch.save(model.cpu().state_dict(), path + '/' + tag + '.pth')
    if validation_loss != None:
        pd.DataFrame({'training_loss': training_loss, 'validation_loss':validation_loss}).to_csv(path + '/' +'training_hist.csv')
    print('Results saved at: \n %s'%path)


def remove_log10(x):
    return torch.pow(10, x)

def transform_to_log10(x):
    return torch.log10(x)

def scale_XYZ(x):
    x[:,0] = x[:,0]/764.431509
    x[:,1] = x[:,1]/785.041607
    x[:,2] = x[:,2]/1083.249944
    return x

def unscale_XYZ(x):
    x[:,0] = 764.431509*x[:,0]
    x[:,1] = 785.041607*x[:,1]
    x[:,2] = 1083.249944*x[:,2]
    return x


# Configurations
torch.multiprocessing.set_sharing_strategy('file_system')

# Constants
features = FEATURES.UPGRADE
truth = TRUTH.UPGRADE

# Configuration
def build_model(run_name, device, archive):
    model = torch.load(os.path.join(archive, f"{run_name}.pth"),pickle_module=dill)
    model.to('cuda:%s'%device[0])
    model.eval()
    model.inference()
    return model 

def train_and_predict_on_validation_set(target,selection, database, pulsemap, batch_size, num_workers, n_epochs, device, run_name,archive, train, patience = 5):
    try:
        del truth[truth.index('interaction_time')]
    except ValueError:
        # not found in list
        pass

    print(f"features: {features}")
    print(f"truth: {truth}")

    training_dataloader, validation_dataloader = make_train_validation_dataloader(
        db = database,
        selection =  selection,
        pulsemaps = pulsemap,
        features = features,
        truth = truth,
        batch_size = batch_size,
        num_workers=num_workers,
    )

    # Building model
    detector = IceCubeUpgrade(
        graph_builder=KNNGraphBuilder(nb_nearest_neighbours=8),
    )
    gnn = DynEdge_V3(
        nb_inputs=detector.nb_outputs,
    )
    if target == 'zenith':
        task = ZenithReconstructionWithKappa(
            hidden_size=gnn.nb_outputs,
            target_label=target,
            loss_function=VonMisesFisher2DLoss(),
        )
    elif target == 'energy':
        task = PassOutput1(hidden_size=gnn.nb_outputs, target_label=target, loss_function=LogCoshLoss(), transform_target = transform_to_log10, transform_inference = remove_log10)

    elif target == 'XYZ':
        task = XYZReconstruction(hidden_size=gnn.nb_outputs, target_label=target, loss_function=EuclideanDistance(), transform_target = scale_XYZ, transform_inference = unscale_XYZ)
    elif target == 'track':
        task = BinaryClassificationTask(hidden_size=gnn.nb_outputs,target_label=target,loss_function=BinaryCrossEntropyLoss())
    elif target == 'truth_flag':
        task = BinaryClassificationTask(hidden_size=gnn.nb_outputs,target_labels=target,loss_function=BinaryCrossEntropyLoss())
    else:
        print('task not found')
    model = Model(
        detector=detector,
        gnn=gnn,
        tasks=[task],
        optimizer_class=Adam,
        optimizer_kwargs={'lr': 1e-03, 'eps': 1e-03},
        scheduler_class=PiecewiseLinearLR,
        scheduler_kwargs={
            'milestones': [0, len(training_dataloader) / 2, len(training_dataloader) * n_epochs],
            'factors': [1e-2, 1, 1e-02],
        },
        scheduler_config={
            'interval': 'step',
        },
     )

    # Training model
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
        ),
        ProgressBar(),
    ]

    if train:
        trainer = Trainer(
            default_root_dir=archive,
            gpus=device,
            max_epochs=n_epochs,
            callbacks=callbacks,
            log_every_n_steps=1,
            logger=None,
        )

        try:
            trainer.fit(model, training_dataloader, validation_dataloader)
        except KeyboardInterrupt:
            print("[ctrl+c] Exiting gracefully.")
            pass

        # Saving model
        model.save(os.path.join(archive, f"{run_name}.pth"))
        model.save_state_dict(os.path.join(archive, f"{run_name}_state_dict.pth"))
        predict(model,trainer,target,selection, database, pulsemap, batch_size, num_workers, n_epochs, device)
    #else:
    #    model.load_state_dict(os.path.join(archive, f"{run_name}_state_dict.pth"))
    #    predict(model,trainer,target,selection, database, pulsemap, batch_size, num_workers, n_epochs, device)


def predict(model,trainer,target,selection, database, pulsemap, batch_size, num_workers, n_epochs, device):
    try:
        del truth[truth.index('interaction_time')]
    except ValueError:
        # not found in list
        pass
    device = 'cuda:%s'%device[0]
    model.to(device)
    _, validation_dataloader = make_train_validation_dataloader(
        db = database,
        selection =  selection,
        pulsemaps = pulsemap,
        features = features,
        truth = truth,
        batch_size = batch_size,
        num_workers=num_workers,
    )
    if target in ['zenith', 'azimuth']:
        #predictor_valid = Predictor(
        #    dataloader=validation_dataloader, 
        #    target=target, 
        #    device=device, 
        #    output_column_names=[target + '_pred', target + '_kappa'],
        #)
        results = get_predictions(
        trainer,
        model,
        validation_dataloader,
        [target + '_pred', target + '_kappa'],
        [target, 'event_no', 'energy'],
        )

    if target in ['track', 'neutrino']:
        #predictor_valid = Predictor(
        #    dataloader=validation_dataloader, 
        #    target=target, 
        #    device=device, 
        #    output_column_names=[target + '_pred'],
        #)
        results = get_predictions(
        trainer,
        model,
        validation_dataloader,
        [target + '_pred'],
        [target, 'event_no', 'energy'],
    )

    if target == 'energy':
        #predictor_valid = Predictor(
        #    dataloader=validation_dataloader, 
        #    target=target, 
        #    device=device, 
        #    output_column_names=[target + '_pred'],
        #    post_processing_method= remove_log10
        #)
        results = get_predictions(
        trainer,
        model,
        validation_dataloader,
        [target + '_pred'],
        [target, 'event_no'],
        )
    if target == 'XYZ':
        #predictor_valid = Predictor(
        #    dataloader=validation_dataloader, 
        #    target=target, 
        #    device=device, 
        #    output_column_names=['position_x_pred','position_y_pred','position_z_pred'],#,'interaction_time_pred'],
        #    post_processing_method= rescale_XYZ
        #)
        results = get_predictions(
        trainer,
        model,
        validation_dataloader,
        ['position_x_pred','position_y_pred','position_z_pred'],
        ['position_x','position_y','position_z', 'event_no', 'energy'],
    )
    save_results(database, run_name, results, archive,model)
    return
# Main function call
if __name__ == "__main__":
    # Run management
    archive = "/groups/icecube/qgf305/graphnet_user/results"
    targets = ['truth_flag']#['zenith' ,'track' , 'energy'] #, 'vertex'] #, 'XYZ']
    batch_size = 1024
    database ='/groups/icecube/asogaard/data/sqlite/dev_step4_numu_140021_second_run/data/dev_step4_numu_140021_second_run.db'
    device = [1]
    n_epochs = 45
    num_workers = 40
    patience = 5
    pulsemap = 'SplitInIcePulses_GraphSage_Pulses'
    # Common variables
    for target in targets:
        selection = get_equal_proportion_neutrino_indices(database)
        #selection = pd.read_csv('/mnt/scratch/rasmus_orsoe/databases/dev_step4_numu_140021_second_run/selection/over10pulses.csv')['event_no'].values.ravel().tolist()

        run_name = "noise_cleaning_{}_GraphSagePulses".format(target)
        
        train_and_predict_on_validation_set(target, selection, database, pulsemap, batch_size, num_workers, n_epochs, device, run_name,archive, train = True)
        #train_and_predict_on_validation_set(target,selection, database, pulsemap, batch_size, num_workers, n_epochs, device, run_name,archive, train = False)
        #predict(target,selection, database, pulsemap, batch_size, num_workers, n_epochs, device)
