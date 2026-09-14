#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use arboard::Clipboard;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use eframe::egui;
use std::sync::{Arc, Mutex};
use tts::Tts;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

#[derive(PartialEq)]
enum AppMode {
    Reader,
    Dictate,
}

struct VoiceReaderApp {
    tts: Tts,
    clipboard: Clipboard,
    speed: f32,
    alpha: f32,

    sentences: Vec<String>,
    current_idx: usize,
    is_playing: bool,
    was_speaking: bool,
    first_frame: bool,

    // App Mode & Dictation State
    mode: AppMode,
    dictation_text: String,
    
    // Audio Capture State
    is_recording: bool,
    is_transcribing: bool,
    audio_buffer: Arc<Mutex<Vec<f32>>>,
    audio_stream: Option<cpal::Stream>,
    
    // The Whisper AI Brain
    whisper_ctx: Option<WhisperContext>,
}

impl VoiceReaderApp {
    fn new() -> Self {
        let mut tts = Tts::default().expect("Failed to initialize TTS");
        if let Ok(voices) = tts.voices() {
            for voice in voices {
                let name = voice.name().to_lowercase();
                if name.contains("zira") || name.contains("female") {
                    let _ = tts.set_voice(&voice);
                    break;
                }
            }
        }
        let default_rate = tts.normal_rate();
        let _ = tts.set_rate(default_rate);

        let whisper_ctx = WhisperContext::new_with_params(
            "ggml-tiny.en.bin",
            WhisperContextParameters::default(),
        ).ok(); 

        Self {
            tts,
            clipboard: Clipboard::new().expect("Failed to bind clipboard"),
            speed: default_rate,
            alpha: 1.0,
            sentences: Vec::new(),
            current_idx: 0,
            is_playing: false,
            was_speaking: false,
            first_frame: true,
            mode: AppMode::Reader,
            dictation_text: String::new(),
            
            is_recording: false,
            is_transcribing: false,
            audio_buffer: Arc::new(Mutex::new(Vec::new())),
            audio_stream: None,
            whisper_ctx,
        }
    }

    fn split_into_sentences(text: &str) -> Vec<String> {
        let mut sentences = Vec::new();
        let mut current = String::new();
        for c in text.chars() {
            current.push(c);
            if c == '.' || c == '?' || c == '!' || c == '\n' {
                if !current.trim().is_empty() {
                    sentences.push(current.clone());
                }
                current.clear();
            }
        }
        if !current.trim().is_empty() {
            sentences.push(current);
        }
        sentences
    }

    fn start_recording(&mut self) {
        self.audio_buffer.lock().unwrap().clear();
        let buffer_clone = self.audio_buffer.clone();

        let host = cpal::default_host();
        let device = host.default_input_device().expect("No input device available");
        let config = device.default_input_config().expect("Failed to get default input config");

        let sample_rate = config.sample_rate().0 as f32;
        let channels = config.channels() as usize;

        let stream = device.build_input_stream(
            &config.into(),
            move |data: &[f32], _: &_| {
                let mut buffer = buffer_clone.lock().unwrap();
                let ratio = sample_rate / 16000.0;
                let mut i = 0.0;
                while (i as usize) * channels < data.len() {
                    let mut sum = 0.0;
                    for c in 0..channels {
                        sum += data[(i as usize) * channels + c];
                    }
                    buffer.push(sum / channels as f32); 
                    i += ratio; 
                }
            },
            |err| eprintln!("Audio capture error: {}", err),
            None,
        ).expect("Failed to build audio stream");

        stream.play().expect("Failed to play audio stream");
        self.audio_stream = Some(stream);
        self.is_recording = true;
    }

    fn stop_and_transcribe(&mut self) {
        self.is_recording = false;
        self.is_transcribing = true;
        self.audio_stream = None; 

        let buffer_clone = self.audio_buffer.clone();
        
        if let Some(ctx) = &self.whisper_ctx {
            let mut state = ctx.create_state().expect("Failed to create Whisper state");
            
            let audio_data = {
                let buffer = buffer_clone.lock().unwrap();
                buffer.clone()
            };

            let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
            params.set_language(Some("en"));
            
            state.full(params, &audio_data).expect("Transcription failed");

            let mut transcribed = String::new();
            let num_segments = state.full_n_segments(); 
            for i in 0..num_segments {
                if let Some(segment) = state.get_segment(i) {
                    transcribed.push_str(&segment.to_str_lossy().unwrap());
                }
            }

            // NEW LOGIC: Append text instead of overwriting!
            let clean_text = transcribed.trim();
            if !clean_text.is_empty() {
                // If there's already text, add a space before adding the new words
                if !self.dictation_text.is_empty() && !self.dictation_text.ends_with(' ') {
                    self.dictation_text.push(' ');
                }
                self.dictation_text.push_str(clean_text);
            }
            
        } else {
            self.dictation_text = "Error: ggml-tiny.en.bin not found!".to_string();
        }
        
        self.is_transcribing = false;
    }
}
impl eframe::App for VoiceReaderApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        
        if self.is_transcribing || self.is_recording || self.is_playing {
            ctx.request_repaint();
        }

        if self.first_frame {
            if let Some(monitor_size) = ctx.input(|i| i.viewport().monitor_size) {
                let app_width = 320.0;
                let app_height = 360.0;
                let x = monitor_size.x - app_width - 15.0;  
                let y = monitor_size.y - app_height - 60.0; 
                ctx.send_viewport_cmd(egui::ViewportCommand::OuterPosition(egui::pos2(x, y)));
                self.first_frame = false;
            }
        }

        if self.is_playing {
            let currently_speaking = self.tts.is_speaking().unwrap_or(false);
            if self.was_speaking && !currently_speaking {
                self.current_idx += 1;
                if self.current_idx < self.sentences.len() {
                    let _ = self.tts.speak(&self.sentences[self.current_idx], true);
                    self.was_speaking = false;
                } else {
                    self.is_playing = false;
                    self.was_speaking = false;
                }
            } else if currently_speaking {
                self.was_speaking = true;
            }
        }

        let mut visuals = egui::Visuals::dark();
        visuals.widgets.noninteractive.rounding = egui::Rounding::same(8.0);
        visuals.widgets.inactive.rounding = egui::Rounding::same(8.0);
        visuals.widgets.hovered.rounding = egui::Rounding::same(8.0);
        visuals.widgets.active.rounding = egui::Rounding::same(8.0);
        visuals.slider_trailing_fill = true; 
        ctx.set_visuals(visuals);

        let mut custom_frame = egui::Frame::central_panel(&ctx.style());
        custom_frame.fill = egui::Color32::from_rgba_unmultiplied(27, 27, 27, (self.alpha * 255.0) as u8);
        custom_frame.rounding = egui::Rounding::same(14.0); 

        egui::CentralPanel::default().frame(custom_frame).show(ctx, |ui| {
            
            ui.horizontal(|ui| {
                ui.add_space(5.0);
                ui.selectable_value(&mut self.mode, AppMode::Reader, egui::RichText::new("🗣 Reader").strong());
                ui.selectable_value(&mut self.mode, AppMode::Dictate, egui::RichText::new("🎙 Dictate").strong());
                
                let drag_space = ui.available_width() - 65.0; 
                let drag_resp = ui.allocate_response(egui::vec2(drag_space, 20.0), egui::Sense::click_and_drag());
                if drag_resp.is_pointer_button_down_on() {
                    ctx.send_viewport_cmd(egui::ViewportCommand::StartDrag);
                }

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button(egui::RichText::new(" X ").strong()).on_hover_cursor(egui::CursorIcon::PointingHand).clicked() {
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                    if ui.button(egui::RichText::new(" _ ").strong()).on_hover_cursor(egui::CursorIcon::PointingHand).clicked() {
                        ctx.send_viewport_cmd(egui::ViewportCommand::Minimized(true));
                    }
                });
            });

            ui.add_space(10.0); 

            if self.mode == AppMode::Reader {
                
                ui.vertical_centered(|ui| {
                    let button_size = egui::vec2(ui.available_width() - 20.0, 45.0);

                    let read_btn = egui::Button::new(
                        egui::RichText::new("READ CLIPBOARD").color(egui::Color32::WHITE).size(18.0).strong()
                    ).fill(egui::Color32::from_rgb(76, 175, 80)).min_size(button_size);

                    if ui.add(read_btn).on_hover_cursor(egui::CursorIcon::PointingHand).clicked() {
                        if let Ok(text) = self.clipboard.get_text() {
                            if !text.trim().is_empty() {
                                self.sentences = Self::split_into_sentences(&text);
                                self.current_idx = 0;
                                self.is_playing = true;
                                self.was_speaking = false;
                                let _ = self.tts.stop(); 
                                if !self.sentences.is_empty() {
                                    let _ = self.tts.speak(&self.sentences[0], true); 
                                }
                            }
                        }
                    }
                });

                ui.add_space(10.0);

                ui.horizontal(|ui| {
                    ui.add_space(10.0); 
                    let available_width = ui.available_width() - 10.0; 
                    let spacing = ui.style().spacing.item_spacing.x;
                    let btn_width = (available_width - (spacing * 3.0)) / 4.0;
                    let btn_size = egui::vec2(btn_width, 35.0);

                    let rw_btn = egui::Button::new(egui::RichText::new("<<").color(egui::Color32::WHITE).strong())
                        .fill(egui::Color32::from_rgb(255, 152, 0)).min_size(btn_size);
                    if ui.add(rw_btn).on_hover_cursor(egui::CursorIcon::PointingHand).clicked() {
                        let _ = self.tts.stop();
                        self.current_idx = self.current_idx.saturating_sub(1);
                        self.was_speaking = false;
                        if self.is_playing && self.current_idx < self.sentences.len() {
                            let _ = self.tts.speak(&self.sentences[self.current_idx], true);
                        }
                    }

                    let play_text = if self.is_playing { " | | " } else { " > " };
                    let play_btn = egui::Button::new(egui::RichText::new(play_text).color(egui::Color32::WHITE).strong())
                        .fill(egui::Color32::from_rgb(33, 150, 243)).min_size(btn_size);
                    if ui.add(play_btn).on_hover_cursor(egui::CursorIcon::PointingHand).clicked() {
                        if self.is_playing {
                            let _ = self.tts.stop();
                            self.is_playing = false;
                            self.was_speaking = false;
                        } else if self.current_idx < self.sentences.len() {
                            self.is_playing = true;
                            let _ = self.tts.speak(&self.sentences[self.current_idx], true);
                        }
                    }

                    let stop_btn = egui::Button::new(egui::RichText::new("STOP").color(egui::Color32::WHITE).strong())
                        .fill(egui::Color32::from_rgb(244, 67, 54)).min_size(btn_size);
                    if ui.add(stop_btn).on_hover_cursor(egui::CursorIcon::PointingHand).clicked() {
                        let _ = self.tts.stop();
                        self.is_playing = false;
                        self.was_speaking = false;
                        self.current_idx = 0; 
                    }

                    let ff_btn = egui::Button::new(egui::RichText::new(">>").color(egui::Color32::WHITE).strong())
                        .fill(egui::Color32::from_rgb(255, 152, 0)).min_size(btn_size);
                    if ui.add(ff_btn).on_hover_cursor(egui::CursorIcon::PointingHand).clicked() {
                        let _ = self.tts.stop();
                        if self.current_idx + 1 < self.sentences.len() {
                            self.current_idx += 1;
                        }
                        self.was_speaking = false;
                        if self.is_playing && self.current_idx < self.sentences.len() {
                            let _ = self.tts.speak(&self.sentences[self.current_idx], true);
                        }
                    }
                });

                ui.add_space(25.0);

                ui.vertical_centered(|ui| {
                    ui.label(egui::RichText::new("Speed Control").strong().color(egui::Color32::LIGHT_GRAY));
                    ui.add_space(5.0);
                    let min_rate = self.tts.min_rate();
                    let normal_rate = self.tts.normal_rate();
                    let comfortable_max = (normal_rate * 2.5).clamp(min_rate, self.tts.max_rate());
                    ui.scope(|ui| {
                        ui.set_max_width(220.0); 
                        ui.style_mut().visuals.selection.bg_fill = egui::Color32::from_rgb(0, 238, 255); 
                        if ui.add(egui::Slider::new(&mut self.speed, min_rate..=comfortable_max)).changed() {
                            let _ = self.tts.set_rate(self.speed);
                        }
                    });
                    
                    ui.add_space(15.0);

                    ui.label(egui::RichText::new("Transparency").strong().color(egui::Color32::LIGHT_GRAY));
                    ui.add_space(5.0);
                    ui.scope(|ui| {
                        ui.set_max_width(220.0);
                        ui.style_mut().visuals.selection.bg_fill = egui::Color32::from_rgb(190, 30, 255); 
                        ui.add(egui::Slider::new(&mut self.alpha, 0.3..=1.0));
                    });
                });

            } else {
                
                // --- DICTATE SCREEN ---
                ui.vertical_centered(|ui| {
                    
                    ui.scope(|ui| {
                        ui.style_mut().visuals.extreme_bg_color = egui::Color32::from_rgb(20, 20, 20);
                        
                        egui::ScrollArea::vertical().max_height(140.0).show(ui, |ui| {
                            // Bound directly to your text so you can click in and manually type!
                            ui.add_sized(
                                [ui.available_width() - 20.0, 140.0],
                                egui::TextEdit::multiline(&mut self.dictation_text)
                                    .hint_text("Your transcribed speech will appear here...")
                                    .margin(egui::vec2(10.0, 10.0))
                            );
                        });
                    });

                    ui.add_space(15.0);

                    let button_size = egui::vec2(ui.available_width() - 20.0, 45.0);
                    
                    if self.is_recording {
                        let rec_btn = egui::Button::new(
                            egui::RichText::new("■ STOP RECORDING").color(egui::Color32::WHITE).size(18.0).strong()
                        ).fill(egui::Color32::from_rgb(244, 67, 54)).min_size(button_size); 
                        
                        if ui.add(rec_btn).on_hover_cursor(egui::CursorIcon::PointingHand).clicked() {
                            self.stop_and_transcribe();
                        }
                    } else if self.is_transcribing {
                        let wait_btn = egui::Button::new(
                            egui::RichText::new("... TRANSCRIBING ...").color(egui::Color32::WHITE).size(18.0).strong()
                        ).fill(egui::Color32::from_rgb(158, 158, 158)).min_size(button_size); 
                        ui.add(wait_btn);
                    } else {
                        let rec_btn = egui::Button::new(
                            egui::RichText::new("🎙 START DICTATION").color(egui::Color32::WHITE).size(18.0).strong()
                        ).fill(egui::Color32::from_rgb(233, 30, 99)).min_size(button_size); 
                        
                        if ui.add(rec_btn).on_hover_cursor(egui::CursorIcon::PointingHand).clicked() {
                            self.start_recording();
                        }
                    }

                    ui.add_space(10.0);

                    let copy_btn = egui::Button::new(
                        egui::RichText::new("📋 COPY TO CLIPBOARD").color(egui::Color32::WHITE).size(14.0).strong()
                    ).fill(egui::Color32::from_rgb(33, 150, 243)).min_size(egui::vec2(ui.available_width() - 20.0, 35.0)); 
                    
                    if ui.add(copy_btn).on_hover_cursor(egui::CursorIcon::PointingHand).clicked() {
                        let _ = self.clipboard.set_text(&self.dictation_text);
                    }
                });
            }
        });
    }
}

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([320.0, 360.0]) 
            .with_always_on_top()
            .with_transparent(true)
            .with_decorations(false),
        ..Default::default()
    };

    eframe::run_native(
        "Ai_Reader",
        options,
        Box::new(|_cc| Box::new(VoiceReaderApp::new())),
    )
}