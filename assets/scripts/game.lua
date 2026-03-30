-- game.lua — Example: fire orb mechanic using spectral threshold callback.
--
-- When band 7 (red/near-IR) energy in the world exceeds 0.8,
-- this script considers it "fire conditions" and logs the event.

local fire_active = false

spectral.on_threshold(0, 0, 0, 10.0, 7, 0.8, function(band, energy)
    if not fire_active then
        fire_active = true
        print(string.format("[game.lua] Fire threshold triggered — band %d energy %.3f", band, energy))
    end
end)

function update(dt)
    -- Per-frame logic: reset fire flag if energy drops
    local current = spectral.get_band(0, 0, 0, 7)
    if current < 0.5 then
        fire_active = false
    end
end
