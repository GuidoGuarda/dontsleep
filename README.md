"# dontsleep" 

DontSleep
DontSleep é um aplicativo de visão computacional que usa a câmera para monitorar sinais de sonolência e exaustão do usuário. O projeto detecta o movimento dos olhos e da boca e emite sons de alerta para ajudar o usuário a se manter acordado. Este aplicativo foi desenvolvido utilizando OpenCV, Mediapipe e Pygame para oferecer uma solução simples e interativa de monitoramento de sonolência.

Funcionalidades
Detecção dos Olhos e Boca: O aplicativo detecta quando os olhos estão fechando ou a boca se abre além de um limite, sugerindo sinais de cansaço.
Alerta Sonoro: Sons de suspense são tocados enquanto os olhos do usuário estão abertos, e um grito é emitido quando a boca é aberta além do limite, simulando um "susto" para alertar o usuário.
Tecnologias Utilizadas
Python: Linguagem de programação principal.
OpenCV: Utilizada para acessar a câmera e processar os frames de vídeo.
Mediapipe: Utilizada para identificar pontos faciais específicos como olhos e boca.
Pygame: Utilizada para o controle e reprodução de sons.
