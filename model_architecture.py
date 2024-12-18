import torch
import torch.utils.data.dataloader
import matplotlib.pyplot as plt
import os
import json
from typing import List, Union

class Model (torch.nn.Module):
    def __init__(self,
                 model_plan_path : str | os.PathLike
                ):
        """
        Opis:
            Implementacja Głębokiej sieci konwolucyjnej, do rozpoznawania kwiatków.
            Automatyczne buduje Model z hiperparametrów z pliku .json.

        Atrybuty:
            self.layers : torch.nn.ModuleList
                Przechowuje zbudowanie bloki sieci

        Publiczne Metody:
            forward() :
                implementacja propagacji w przód modelu
            fit() :
                Przeprowadza pętle uczenia CNN.
        
        Prywatne Metody:
            __generate_convolutional_block() :
                Generuje blok konwolucyjny
            __generate_linear_block() :
                Generuje blok klasycznej sieci neuronowej 
        """
        super(Model, self).__init__()

        # Ładujemy nasz model z jsona
        with open(model_plan_path, "r") as file:
            model_arch = json.load(file)

        in_channel : int  = 3
        #layers : List[Union[torch.nn.Sequential, torch.nn.Flatten]]= []
        self.layers = torch.nn.ModuleList()

        # Budujemy bloki konwolucyjne z opisu z json-a
        for description in model_arch["convolutional_layers"]:
            self.layers.extend(self.__generate_convolutional_block(in_channel, **description))
            in_channel = description["out_channels"]

        # To jest chyba najleniwszy sposób by dostać wymiary naszej sieci xddd
        # czytajcie, nie oceniajcie
        in_channel = torch.zeros((1,3,*model_arch["input_size"]))

        for layer in self.layers:
            in_channel  = layer.forward(in_channel)

        in_channel = torch.numel(in_channel)

        self.layers.extend([torch.nn.Flatten()])

        # Budujemy bloki z opisu z json-a
        for description in model_arch["linear_layers"]:
            self.layers.extend(self.__generate_linear_block(in_channel, **description))
            in_channel = description["out_channels"]

        self.layers.extend([torch.nn.Linear(in_channel, 102)])

    def forward(self,
                x: torch.Tensor
                ) -> torch.Tensor:
        
        """
        Opis: 
            implementacja propagacji w przód modelu

        Parametry:
            x : torch.Tensor
                input do naszej sieci.
                Formatu (batch,  liczba kanałów obrazka (3),  wysokość,  szerokość)
        
        Zwraca:
            x : Torch.Tensor
                Output sieci
                Formatu (batch, 102)
        
        """
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def fit(self,
            train_dataset: torch.utils.data.DataLoader,
            test_dataset: torch.utils.data.DataLoader,
            epochs:int = 1,
            learning_rate: float = 0.001,
            model_dir: str | os.PathLike = None,
            ) -> None:
        
        """
        Opis:
            Przeprowadza pętle uczenia CNN.
            Co epoch przeprowadza walidacje modelu.
            Jeżeli funkcja straty (loss) jest najmniejsza w danym treningu to zapisuje model w danym folderze.


        Parametry:
            train_dataset : torch.utils.data.DataLoader
                obiekt reprezentujący nasze dane treningowe
            test_dataset :  torch.utils.data.DataLoader
                obiekt reprezentujący nasze dane walidacyjne
            epochs : int
                ilość cykli w których model uczy się na całym zbiorze danych
            learning_rate : float
                tzw. współczynnik uczenia, decyduje o tym jak mocno zmieniamy wagi podczas uczenia
                Używamy optymilazatora Adam więc ta wartość nie jest stała
            model_dir : str | os.Pathlike
                ścieżka do folderu, w którym chcemy zapisać nasze wagi
                Jeżeli nie jest podana, to model zapisuje się w folderze projektu
        """

        OPTIMIZER = torch.optim.Adam(self.parameters(), learning_rate)
        LOSS_FN = torch.nn.CrossEntropyLoss()

        vloss_history = []
        loss_history = []
        vacc_history = []

        # zmienna do przechowywania najlepszego błędu validacyjnego podczas uczenia
        best_vloss = float("inf")

        for epoch in range(epochs):
            print(f"EPOCH: {epoch+1}/{epochs}")
            
            # tym chciałbym trackować treningowy loss modelu, by potem nawet przesłac to do tensorboard jak bedzie czas to zaimplemetować
            save_loss = 0

            # iterujemy biorąc kolejne batche ze zbioru
            for i, data in enumerate(train_dataset):
                # set training mode
                self.train()
                x, true_labels = data

                # zerujemy gradienty
                OPTIMIZER.zero_grad()

                # propagacja w przód
                predicted_labels = self.forward(x)

                # obliczamy funckje straty
                loss = LOSS_FN(predicted_labels, true_labels)

                # obliczamy gradienty dla wag
                loss.backward()

                # zmieniamy parametry naszego modelu
                OPTIMIZER.step()

                save_loss += loss.item()
                
                # co piąty batch aktualizujemy progress bar z którego jestem o wiele bardziej dumny niż powinienem
                # trochę potworek ale nie ma co analizować
                if i % 5 == 4:
                    print(u"    Batch: {} \u2551{}{}\u2551 Loss: {}".format(
                        str(i+1).zfill(3), # numer batcha
                        u"\u2588" * int((i+1)/5), # wypełniona część progress bara
                        " " * int(len(train_dataset)/5 - (i+1)/5), # pusty progress bar
                        loss.item()), end="\r") # loss

            save_loss /= i+1
            loss_history.append(save_loss)

            # po aktualizacji wag przeprowadzamy walidacje modelu
            # w tym celu wyłączamy śledzenie gradientu by troche oszczędzić na czasie
            with torch.no_grad():
                self.eval()
                vloss = 0.
                vacc = 0.
                for i, v_data in enumerate(test_dataset):
                    v_x, v_true_labels = v_data

                    # propagacja w przód
                    v_predicted_labels = self.forward(v_x)

                    # obliczamy funckje straty
                    vloss += LOSS_FN(v_predicted_labels, v_true_labels).item()

                    vacc += self.accuracy(y_predicted=v_predicted_labels,
                                          y_true=v_true_labels)

                vloss /= (i+1) # enumerate lczy od 0 więc by poprawnie wyliczyć średnią dodajemy +1
                vacc /= (i+1)

                # tu jeżeli zaobserwujemy najmniejszy walidacyjny loss podczas uczenia to zapisujemy model we wskazanym folderze
                if vloss < best_vloss:
                    best_vloss = vloss

                    if model_dir is not None:
                        torch.save(self.state_dict(), os.path.join(model_dir, "model.pt"))
                    else:
                        torch.save(self.state_dict(), "model.pt")

                    message = u" \033[92m BEST MODEL UP TO DATE, MODEL SAVED \033[0m"
                else: 
                    message = u" \033[93m MODEL WORSE THAN PREVIOUS, WASN'T SAVED \033[0m"
                    
                print(f"Loss: {save_loss}   Validation loss: {vloss}    Accuracy: {vacc} \n", message)
                
                vloss_history.append(vloss)
                vacc_history.append(vacc)

        # stwórzmy wykres pokazujący historię uczenia modelu
        fig, (loss_ax, acc_ax) = plt.subplots(2, sharex=True)

        fig.suptitle('Wykresy modelu', fontsize=36)
        fig.set_figheight(10)
        fig.set_figwidth(10)

        loss_ax.plot(vloss_history, label="walidacja")
        loss_ax.plot(loss_history, label="trening")
        loss_ax.set_title("Funkcja straty")
        loss_ax.legend(loc="upper right")
        loss_ax.set_xlabel("epoch")
        loss_ax.set_ylabel("Wartość funkcji straty")

        acc_ax.plot(vacc_history)
        acc_ax.set_title("Dokładność")
        acc_ax.set_xlabel("epoch")
        acc_ax.set_ylabel("Wartość dokładności")
        acc_ax.set_yticks([0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

        fig.savefig("plots/model_plot.png")


    def __generate_convolutional_block(self, 
                                     in_channels : int,
                                     out_channels : int,
                                     conv_kernel_size : int,
                                     conv_stride : int,
                                     padding : int,
                                     pool_kernel_size : int,
                                     pool_stride : int,
                                     dropout_rate : int
                                    ) -> torch.nn.Sequential:
        """
        Opis:
            Generuje blok konwolucyjny składający się z 
                1) warstwy konwolucyjnej
                2) max polling
                3) normalizacji wsadowej
                4) funkcji aktywacji ReLU
        
        Parametry:
            Parametrami funkcji są hiperparametry modelu
        """

        block = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, conv_kernel_size, conv_stride, padding),
            torch.nn.MaxPool2d(pool_kernel_size, pool_stride),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.Dropout2d(dropout_rate),
            torch.nn.ReLU()
        )

        return block
    
    def __generate_linear_block(self,
                                in_channels : int,
                                out_channels : int,
                                dropout_rate : int
                                ) -> torch.nn.Sequential:
        """
        Opis:
            Generuje blok klasycznej sieci neuronowej składający się z 
                1) warstwy liniowej
                2) normalizacji wsadowej
                3) funkcji aktywacji ReLU
        
        Parametry:
            Parametrami funkcji są hiperparametry modelu
        """

        block = torch.nn.Sequential(
            torch.nn.Linear(in_channels, out_channels),
            torch.nn.BatchNorm1d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate)
        )

        return block
    
    def accuracy(self,
                   y_predicted :  torch.Tensor,
                   y_true : torch.Tensor
                  ) -> float:
        """
        Opis:
            Oblicza dokładność dla dwóch tensorów
        
        Parametry :
            y_predicted : torch.Tensor
                tensor będący predykcjami naszego modelu postaci (batch, 102)
            y_true : torch.Tensor
                tensor będący prawdzimy wartościami naszych klas

        Zwraca:
            Wartość dokładności w przedziale [0.0, 1.0]
        """

        #assert torch.shape(y_predicted)[0] == torch.shape(y_true)[0], "Podane tensory do siebie nie pasują"
        y_predicted = torch.argmax(y_predicted, dim=1)
        return torch.sum(torch.eq(y_true, y_predicted).float()) / len(y_true)

        



