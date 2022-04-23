FROM dealii/dealii:latest
# UPGRADE THE SYSTEM
RUN sudo apt update
RUN sudo apt upgrade -y
# INSTALL NEOVIM
RUN wget -O nvim.deb https://github.com/neovim/neovim/releases/download/v0.7.0/nvim-linux64.deb
RUN sudo apt install -y ./nvim.deb
RUN mkdir .config
RUN rm nvim.deb
# INSTALL POWERSHELL
RUN sudo apt-get install -y wget apt-transport-https software-properties-common
RUN wget -q https://packages.microsoft.com/config/ubuntu/20.04/packages-microsoft-prod.deb
RUN sudo dpkg -i packages-microsoft-prod.deb
RUN sudo apt-get update
RUN sudo apt-get install -y powershell
# INSTALL NODE
RUN curl -fsSL https://deb.nodesource.com/setup_17.x | sudo -E bash -
RUN sudo apt install -y nodejs
# SET WORKING ENVIRONMENT
WORKDIR /home/dealii
CMD ["pwsh"]
EXPOSE 3000
