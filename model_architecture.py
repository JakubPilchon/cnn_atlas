import torch
import torch.utils.data.dataloader
import os

class Model (torch.nn.Module):
    def __init__(self):
        """
        Opis:
            Implementacja Głębokiej sieci konwolucyjnej, do rozpoznawania kwiatkóq

        Metody:
            forward :
                implementacja propagacji w przód modelu
            fit :
                Przeprowadza pętle uczenia CNN.
        """
        super(Model, self).__init__()

        ## MODEL ARCHITECTURE
        self.activation = torch.nn.ReLU()
        
        self.conv_1 = torch.nn.Conv2d(in_channels=3, out_channels=32,kernel_size=5, stride=3, padding=1)
        self.padd_1 = torch.nn.MaxPool2d(kernel_size= 2,stride=2)
        self.norm_1 = torch.nn.BatchNorm2d(32)

        self.conv_2 = torch.nn.Conv2d(in_channels=32, out_channels=64,kernel_size=5, stride=3, padding=1)
        self.padd_2 = torch.nn.MaxPool2d(kernel_size= 2,stride=2)
        self.norm_2 = torch.nn.BatchNorm2d(64)

        self.conv_3 = torch.nn.Conv2d(in_channels=64, out_channels=128,kernel_size=3, stride=1, padding=1)
        self.padd_3 = torch.nn.MaxPool2d(kernel_size= 2,stride=2)
        self.norm_3 = torch.nn.BatchNorm2d(128)

        self.flatten = torch.nn.Flatten()
        
        self.dense_1 = torch.nn.Linear(128*6*8, 2048)
        
        self.dense_2 = torch.nn.Linear(2048, 512)

        self.dense_3 = torch.nn.Linear(512, 102)

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
        ## IMPLEMENT FORWARD PASS HERE
        x = self.conv_1(x)
        x = self.padd_1(x)
        x = self.norm_1(x)
        x = self.activation(x)

        x = self.conv_2(x)
        x = self.padd_2(x)
        x = self.norm_2(x)
        x = self.activation(x)

        x = self.conv_3(x)
        x = self.padd_3(x)
        x = self.norm_3(x)
        x = self.activation(x)

        x = self.flatten(x)
       
        x = self.dense_1(x)
        x = self.activation(x)

        x = self.dense_2(x)
        x = self.activation(x)

        x = self.dense_3(x)
        
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

        # zmienna do przechowywania najlepszego błędu validacyjnego podczas uczenia
        best_vloss = float("inf")

        for epoch in range(epochs):
            print(f"EPOCH: {epoch}/{epochs}")
            
            # tym chciałbym trackować treningowy loss modelu, by potem nawet przesłac to do tensorboard jak bedzie czas to zaimplemetować
            save_loss = 0

            # iterujemy biorąc kolejne batche ze zbioru
            for i, data in enumerate(train_dataset):
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

            # po aktualizacji wag przeprowadzamy walidacje modelu
            # w tym celu wyłączamy śledzenie gradientu by troche oszczędzić na czasie
            with torch.no_grad():
                vloss = 0.
                for i, v_data in enumerate(test_dataset):
                    v_x, v_true_labels = v_data

                    # propagacja w przód
                    v_predicted_labels = self.forward(v_x)

                    # obliczamy funckje straty
                    vloss += LOSS_FN(v_predicted_labels, v_true_labels).item()

                vloss /= (i+1) # enumerate lczy od 0 więc by poprawnie wyliczyć średnią dodajemy +1

                # tu jeżeli zaobserwujemy najmniejszy walidacyjny loss podczas uczenia to zapisujemy model we wskazanym folderze
                if vloss < best_vloss:
                    best_vloss = vloss

                    if model_dir is not None:
                        torch.save(self.state_dict(), os.path.join(model_dir, "model.pt"))
                    else:
                        torch.save(self.state_dict(), "model.pt")

                    print(f"Validation loss: {vloss}", u" \033[92m BEST MODEL UP TO DATE, MODEL SAVED \033[0m")
                else: 
                    print(f"Validation loss: {vloss}", u" \033[93m MODEL WORSE THAN PREVIOUS, WASN'T SAVED \033[0m")



