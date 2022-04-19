import gradio as gr

from libreai.utils import Args, text2img


def run(prompt, steps, width, height, images, scale):
    opt = Args()
    # "the prompt to render"
    opt.prompt = prompt
    opt.ddim_eta = 0.0  # "ddim eta (eta=0.0 corresponds to deterministic sampling"
    opt.n_samples = int(images)  # "how many samples to produce for the given prompt"
    opt.n_iter = 4  # "sample this often"
    opt.scale = scale  # "unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))"
    opt.ddim_steps = int(steps)  # "number of ddim sampling steps"
    opt.plms = True  # "use plms sampling: much faster"
    opt.is_notebook = True  # "if we are on a notebook or not"
    opt.outdir = "outputs/txt2img-samples"  # "dir to write results to"
    opt.H = int(height)  # "image height, in pixel space"
    opt.W = int(width)  # "image width, in pixel space"

    grid_image, all_samples_images = text2img(opt)

    return (grid_image, all_samples_images, None)


image = gr.outputs.Image(type="pil", label="Your result")
css = ".output-image{height: 528px !important} .output-carousel .output-image{height:272px !important} a{text-decoration: underline}"
iface = gr.Interface(fn=run, inputs=[
    gr.inputs.Textbox(label="Prompt - what to paint?",default="robot self portrait, oil in canvas"),
    gr.inputs.Slider(label="Steps - more steps can increase quality but will take longer to generate",default=45,maximum=50,minimum=1,step=1),
    gr.inputs.Radio(label="Width", choices=[32,64,128,256],default=256),
    gr.inputs.Radio(label="Height", choices=[32,64,128,256],default=256),
    gr.inputs.Slider(label="Images - How many images you wish to generate", default=2, step=1, minimum=1, maximum=4),
    gr.inputs.Slider(label="Diversity scale - How different from one another you wish the images to be",default=5.0, minimum=1.0, maximum=15.0),
    #gr.inputs.Slider(label="ETA - between 0 and 1. Lower values can provide better quality, higher values can be more diverse",default=0.0,minimum=0.0, maximum=1.0,step=0.1),
    ],
    outputs=[image,gr.outputs.Carousel(label="Individual images",components=["image"]),gr.outputs.Textbox(label="Error")],
    css=css,
    title="[EPIPHANY] - Text to Image using Latent Diffusion",
    description="",
    article="",
    allow_flagging="never"
    )
iface.launch(enable_queue=True)