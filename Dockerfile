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
RUN wget "https://nodejs.org/dist/v18.0.0/node-v18.0.0-linux-x64.tar.xz" -O "./node-v18.0.0-linux-x64.tar.xz"
RUN sudo mkdir -p /usr/local/lib/nodejs
RUN sudo tar -xJvf "node-v18.0.0-linux-x64.tar.xz" -C /usr/local/lib/nodejs
RUN rm node-v18.0.0-linux-x64.tar.xz
RUN echo "export PATH=/usr/local/lib/nodejs/node-v18.0.0-linux-x64/bin:$PATH" >> ~/.bashrc
# SET WORKING ENVIRONMENT
WORKDIR /home/dealii
CMD ["bash"]
EXPOSE 3000
