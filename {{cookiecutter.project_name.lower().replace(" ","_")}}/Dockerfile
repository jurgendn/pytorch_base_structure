FROM gr000a1/torch-gpu:0.4.1

RUN apt-get update
RUN pip3 install pip --upgrade

# Install requirements
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Switch to non-root user
RUN useradd -m working
USER working
WORKDIR /home/working

# Execute
# ENTRYPOINT [ "executable" ]
CMD ["bash", "./.docker/startup.sh"]